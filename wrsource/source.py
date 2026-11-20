"""Versioned, revisionable weighted record source backed by SQLite.

Design summary
--------------

* ``rows`` holds the *current* mutable state of every row ever admitted.  A
  row keeps a stable ``row_id`` for its whole life.  ``rev`` increases by one
  per committed batch in which the row's logical content changes (group,
  weight, body, active flag).  A shard move is physical only: it updates
  ``rows.shard`` without a new revision and without a new source version.
* ``versions`` / ``version_rows`` are immutable full snapshots of the active
  set.  A new version is published only when a batch changes logical content.
  Snapshots contain group, weight, body digest and revision -- never the
  shard, so moving a row between shards cannot alter row identity or content
  inside an existing version.
* ``bodies`` is content-addressed and append-only.  Every published version
  therefore keeps readable bodies even after later corrections.  Old bodies
  are never garbage collected in this implementation.
* Pagination is keyed by ``(version_id, row_id)``.  Because a version is a
  frozen snapshot, writers publishing newer versions cannot make an existing
  reader skip or duplicate a row.  UTF-8 binary ordering coincides with
  Unicode code-point ordering, and the sampler uses Python string ordering;
  both agree.
* A batch is staged in memory, collapsed to each row's *final* state, then
  written and snapshot in one ``BEGIN IMMEDIATE`` transaction.  Any error
  rolls the whole batch back.

Run ``python -m wrsource.source --help`` for the command line.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 1
MAX_WEIGHT = 100
MIN_WEIGHT = 1

OP_KINDS = {"add", "set_group", "set_weight", "set_body", "set_shard", "retract"}


class SourceError(Exception):
    pass


class BatchError(SourceError):
    pass


class VersionError(SourceError):
    pass


_JSON_SEP = (",", ":")


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=_JSON_SEP,
                      ensure_ascii=False)


def _check_text(value, what, allow_empty=False):
    if not isinstance(value, str):
        raise BatchError(f"{what} must be a string")
    if not allow_empty and not value:
        raise BatchError(f"{what} must not be empty")
    if "\x00" in value:
        raise BatchError(f"{what} must not contain NUL")
    return value


def _check_weight(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BatchError("weight must be an integer")
    if not MIN_WEIGHT <= value <= MAX_WEIGHT:
        raise BatchError(f"weight must be in {MIN_WEIGHT}..{MAX_WEIGHT}")
    return value


def _check_body(value) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BatchError("body bytes are not valid UTF-8") from exc
    return _check_text(value, "body", allow_empty=True)


@dataclass
class _Stage:
    row_id: str
    in_db: bool
    orig: Optional[sqlite3.Row]
    group: str
    weight: int
    body_sha256: str
    active: bool
    shard: str
    created: bool = False

    def logical_tuple(self):
        return (self.active, self.group, self.weight, self.body_sha256)


@dataclass
class BatchResult:
    version: Optional[int]
    created: bool
    logical_row_ids: List[str] = field(default_factory=list)
    shard_row_ids: List[str] = field(default_factory=list)
    discarded_row_ids: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "version_created": self.created,
            "logical_row_ids": sorted(self.logical_row_ids),
            "shard_row_ids": sorted(self.shard_row_ids),
            "discarded_row_ids": sorted(self.discarded_row_ids),
        }


class SourceService:
    """A revisionable weighted record source.

    All methods are safe to call from independent clients: every call opens
    its own SQLite connection in WAL mode.
    """

    def __init__(self, path: str):
        self.path = path
        self._initialised = False

    # -- connections -----------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def init(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS meta(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS bodies(
                    sha256 TEXT PRIMARY KEY,
                    data BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rows(
                    row_id TEXT PRIMARY KEY,
                    group_name TEXT NOT NULL,
                    weight INTEGER NOT NULL
                        CHECK(weight BETWEEN 1 AND 100),
                    body_sha256 TEXT NOT NULL,
                    rev INTEGER NOT NULL,
                    active INTEGER NOT NULL,
                    shard TEXT NOT NULL,
                    FOREIGN KEY(body_sha256) REFERENCES bodies(sha256)
                );
                CREATE TABLE IF NOT EXISTS versions(
                    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    row_count INTEGER NOT NULL,
                    note TEXT
                );
                CREATE TABLE IF NOT EXISTS version_rows(
                    version_id INTEGER NOT NULL
                        REFERENCES versions(version_id),
                    row_id TEXT NOT NULL,
                    group_name TEXT NOT NULL,
                    weight INTEGER NOT NULL,
                    body_sha256 TEXT NOT NULL,
                    rev INTEGER NOT NULL,
                    PRIMARY KEY(version_id, row_id),
                    FOREIGN KEY(body_sha256) REFERENCES bodies(sha256)
                );
                CREATE INDEX IF NOT EXISTS ix_vr_order
                    ON version_rows(version_id, row_id);
                CREATE INDEX IF NOT EXISTS ix_vr_group
                    ON version_rows(version_id, group_name, row_id);
                CREATE TABLE IF NOT EXISTS batch_log(
                    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id INTEGER,
                    created_at REAL NOT NULL,
                    note TEXT,
                    ops_json TEXT NOT NULL,
                    result_json TEXT NOT NULL
                );
                COMMIT;
                """
            )
            # Publish an explicit empty version 0 so every reader always has a
            # fixed version to bind to and the first productive batch is 1.
            row = conn.execute("SELECT COUNT(*) AS n FROM versions").fetchone()
            if row["n"] == 0:
                conn.execute(
                    "INSERT INTO versions(version_id, created_at, row_count, "
                    "note) VALUES (0, ?, 0, ?)", (time.time(),
                                                  "initial empty version"))
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(SCHEMA_VERSION),))
            self._initialised = True
        finally:
            conn.close()

    def _ensure(self):
        if not self._initialised:
            self.init()

    # -- batch application ----------------------------------------------
    def apply_batch(self, ops: Sequence[dict], note: Optional[str] = None
                    ) -> BatchResult:
        """Apply a batch atomically, collapsing each row to final state.

        Supported ops (plain dicts, JSON friendly)::

            {"op":"add","row_id":..,"group":..,"weight":..,"body":..,"shard":..}
            {"op":"set_group","row_id":..,"group":..}
            {"op":"set_weight","row_id":..,"weight":..}
            {"op":"set_body","row_id":..,"body":..}
            {"op":"set_shard","row_id":..,"shard":..}
            {"op":"retract","row_id":..}
        """
        self._ensure()
        if not isinstance(ops, (list, tuple)):
            raise BatchError("ops must be a list")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                result = self._apply_batch_tx(conn, ops, note)
                conn.execute("COMMIT")
                return result
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

    def _apply_batch_tx(self, conn: sqlite3.Connection, ops: Sequence[dict],
                        note: Optional[str]) -> BatchResult:
        stage: Dict[str, _Stage] = {}
        order: List[str] = []

        def load(row_id: str) -> _Stage:
            rid = _check_text(row_id, "row_id")
            if rid in stage:
                return stage[rid]
            row = conn.execute(
                "SELECT row_id, group_name, weight, body_sha256, rev, active, "
                "shard FROM rows WHERE row_id=?", (rid,)).fetchone()
            if row is None:
                raise BatchError(f"row_id not found: {rid!r}")
            st = _Stage(rid, True, row, row["group_name"], row["weight"],
                        row["body_sha256"], bool(row["active"]), row["shard"])
            stage[rid] = st
            order.append(rid)
            return st

        def store_body(body_text: str) -> str:
            import hashlib
            data = body_text.encode("utf-8")
            digest = hashlib.sha256(data).hexdigest()
            conn.execute(
                "INSERT INTO bodies(sha256, data) VALUES (?, ?) "
                "ON CONFLICT(sha256) DO NOTHING", (digest, data))
            return digest

        for raw in ops:
            if not isinstance(raw, dict):
                raise BatchError("each op must be a dict")
            op = raw.get("op")
            if op not in OP_KINDS:
                raise BatchError(f"unknown op: {op!r}")
            rid = _check_text(raw.get("row_id"), "row_id")

            if op == "add":
                if rid in stage and stage[rid].active:
                    raise BatchError(f"duplicate add for {rid!r} in batch")
                if rid in stage:
                    # Retracted earlier in this batch: re-add is legal; only
                    # the batch's final state matters.
                    prev = stage[rid]
                    orig = prev.orig
                else:
                    prev = None
                    orig = conn.execute(
                        "SELECT row_id, group_name, weight, body_sha256, rev, "
                        "active, shard FROM rows WHERE row_id=?", (rid,)
                    ).fetchone()
                if orig is not None and bool(orig["active"]) and prev is None:
                    raise BatchError(f"row already active: {rid!r}")
                st = _Stage(rid, orig is not None, orig, "", 1, "", True, "",
                            created=orig is None)
                st.group = _check_text(raw.get("group"), "group")
                st.weight = _check_weight(raw.get("weight"))
                st.body_sha256 = store_body(_check_body(raw.get("body")))
                st.shard = _check_text(raw.get("shard"), "shard")
                st.active = True
                if prev is None:
                    order.append(rid)
                stage[rid] = st
            elif op == "set_group":
                st = load(rid)
                if not st.active:
                    raise BatchError(f"cannot regroup inactive row {rid!r}")
                st.group = _check_text(raw.get("group"), "group")
            elif op == "set_weight":
                st = load(rid)
                if not st.active:
                    raise BatchError(f"cannot reweight inactive row {rid!r}")
                st.weight = _check_weight(raw.get("weight"))
            elif op == "set_body":
                st = load(rid)
                if not st.active:
                    raise BatchError(f"cannot edit inactive row {rid!r}")
                st.body_sha256 = store_body(_check_body(raw.get("body")))
            elif op == "set_shard":
                st = load(rid)
                st.shard = _check_text(raw.get("shard"), "shard")
            elif op == "retract":
                st = load(rid)
                st.active = False

        # Collapse every touched row to its final state.  A logical change
        # bumps ``rev`` and publishes a new version; a pure shard move keeps
        # the same rev and the same version.
        logical: List[str] = []
        shard_moves: List[str] = []
        discarded: List[str] = []
        upserts: List[Tuple[_Stage, int]] = []

        for rid in order:
            st = stage[rid]
            if st.created and not st.active:
                # Added and retracted inside one batch: nothing ever existed.
                discarded.append(rid)
                continue
            if st.active:
                _check_weight(st.weight)
                _check_text(st.group, "group")
                _check_text(st.shard, "shard")

            if st.orig is None:
                rev = 1
                logical.append(rid)
                upserts.append((st, rev))
                continue

            old_logical = (bool(st.orig["active"]), st.orig["group_name"],
                           st.orig["weight"], st.orig["body_sha256"])
            if st.logical_tuple() != old_logical:
                rev = st.orig["rev"] + 1
                logical.append(rid)
                upserts.append((st, rev))
            elif st.shard != st.orig["shard"]:
                shard_moves.append(rid)
                upserts.append((st, st.orig["rev"]))

        for st, rev in upserts:
            if st.orig is None:
                conn.execute(
                    "INSERT INTO rows(row_id, group_name, weight, body_sha256, "
                    "rev, active, shard) VALUES (?,?,?,?,?,?,?)",
                    (st.row_id, st.group, st.weight, st.body_sha256, rev, 1,
                     st.shard))
            else:
                conn.execute(
                    "UPDATE rows SET group_name=?, weight=?, body_sha256=?, "
                    "rev=?, active=?, shard=? WHERE row_id=?",
                    (st.group, st.weight, st.body_sha256, rev,
                     1 if st.active else 0, st.shard, st.row_id))

        created = bool(logical)
        if created:
            cur = conn.execute(
                "INSERT INTO versions(created_at, row_count, note) "
                "VALUES (?, 0, ?)", (time.time(), note))
            version_id = int(cur.lastrowid)
            conn.execute(
                "INSERT INTO version_rows(version_id, row_id, group_name, "
                "weight, body_sha256, rev) "
                "SELECT ?, row_id, group_name, weight, body_sha256, rev "
                "FROM rows WHERE active=1", (version_id,))
            count = conn.execute(
                "SELECT COUNT(*) AS n FROM version_rows WHERE version_id=?",
                (version_id,)).fetchone()["n"]
            conn.execute("UPDATE versions SET row_count=? WHERE version_id=?",
                         (count, version_id))
        else:
            row = conn.execute(
                "SELECT MAX(version_id) AS v FROM versions").fetchone()
            version_id = row["v"]

        result = BatchResult(version_id, created, logical, shard_moves,
                             discarded)
        conn.execute(
            "INSERT INTO batch_log(version_id, created_at, note, ops_json, "
            "result_json) VALUES (?,?,?,?,?)",
            (version_id if created else None, time.time(), note,
             json.dumps(list(ops), ensure_ascii=False),
             json.dumps(result.as_dict(), ensure_ascii=False)))
        return result

    # -- version metadata ------------------------------------------------
    def list_versions(self) -> List[dict]:
        self._ensure()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT version_id, created_at, row_count, note FROM versions "
                "ORDER BY version_id").fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def latest_version(self) -> int:
        self._ensure()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT MAX(version_id) AS v FROM versions").fetchone()
            if row["v"] is None:
                raise VersionError("no version published")
            return int(row["v"])
        finally:
            conn.close()

    def get_version(self, version_id: int) -> dict:
        self._ensure()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT version_id, created_at, row_count, note FROM versions "
                "WHERE version_id=?", (version_id,)).fetchone()
            if row is None:
                raise VersionError(f"unknown version: {version_id}")
            return dict(row)
        finally:
            conn.close()

    def _require_version(self, conn, version_id: int) -> None:
        row = conn.execute(
            "SELECT 1 FROM versions WHERE version_id=?", (version_id,)
        ).fetchone()
        if row is None:
            raise VersionError(f"unknown version: {version_id}")

    # -- reads at a fixed version ---------------------------------------
    _ROW_SELECT = """
        SELECT vr.row_id, vr.group_name AS "group", vr.weight,
               vr.body_sha256, vr.rev, r.shard, b.data AS body_blob
        FROM version_rows vr
        JOIN rows r ON r.row_id = vr.row_id
        JOIN bodies b ON b.sha256 = vr.body_sha256
        WHERE vr.version_id = ?
    """

    @staticmethod
    def _row_dict(row: sqlite3.Row) -> dict:
        return {
            "row_id": row["row_id"],
            "group": row["group"],
            "weight": row["weight"],
            "rev": row["rev"],
            "shard": row["shard"],
            "body_sha256": row["body_sha256"],
            "body": bytes(row["body_blob"]).decode("utf-8"),
        }

    def read_page(self, version_id: int, after_row_id: Optional[str] = None,
                  limit: int = 1000) -> dict:
        """One page of a fixed version, ordered by row_id code points."""
        self._ensure()
        if limit <= 0:
            raise ValueError("limit must be positive")
        if after_row_id is not None:
            _check_text(after_row_id, "after_row_id", allow_empty=True)
        conn = self._connect()
        try:
            self._require_version(conn, version_id)
            sql = self._ROW_SELECT
            params: List[object] = [version_id]
            if after_row_id is not None:
                sql += " AND vr.row_id > ?"
                params.append(after_row_id)
            sql += " ORDER BY vr.row_id LIMIT ?"
            params.append(limit + 1)
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        has_more = len(rows) > limit
        page = [self._row_dict(r) for r in rows[:limit]]
        last = page[-1]["row_id"] if page else after_row_id
        cursor = None
        if has_more and last is not None:
            cursor = encode_cursor(version_id, last)
        return {
            "version": version_id,
            "rows": page,
            "next_cursor": cursor,
            "done": not has_more,
        }

    def open_reader(self, version_id: int, page_size: int = 1000
                    ) -> "VersionReader":
        return VersionReader(self, version_id, page_size)

    def read_all(self, version_id: int, page_size: int = 5000) -> List[dict]:
        return list(self.open_reader(version_id, page_size))

    def all_rows_meta(self, version_id: int) -> List[dict]:
        """All active rows of a version, without body bytes."""
        self._ensure()
        conn = self._connect()
        try:
            self._require_version(conn, version_id)
            rows = conn.execute(
                "SELECT vr.row_id, vr.group_name AS \"group\", vr.weight, "
                "vr.body_sha256, vr.rev, r.shard "
                "FROM version_rows vr JOIN rows r ON r.row_id=vr.row_id "
                "WHERE vr.version_id=? ORDER BY vr.row_id",
                (version_id,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def group_rows_meta(self, version_id: int, group: str) -> List[dict]:
        self._ensure()
        conn = self._connect()
        try:
            self._require_version(conn, version_id)
            rows = conn.execute(
                "SELECT vr.row_id, vr.group_name AS \"group\", vr.weight, "
                "vr.body_sha256, vr.rev, r.shard "
                "FROM version_rows vr JOIN rows r ON r.row_id=vr.row_id "
                "WHERE vr.version_id=? AND vr.group_name=? "
                "ORDER BY vr.row_id", (version_id, group)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_row(self, version_id: int, row_id: str) -> Optional[dict]:
        self._ensure()
        conn = self._connect()
        try:
            self._require_version(conn, version_id)
            row = conn.execute(
                self._ROW_SELECT + " AND vr.row_id=?", (version_id, row_id)
            ).fetchone()
            return None if row is None else self._row_dict(row)
        finally:
            conn.close()

    def get_rows(self, version_id: int, row_ids: Iterable[str]) -> Dict[str,
                                                                       dict]:
        ids = list(dict.fromkeys(row_ids))
        out: Dict[str, dict] = {}
        if not ids:
            return out
        self._ensure()
        conn = self._connect()
        try:
            self._require_version(conn, version_id)
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                q = self._ROW_SELECT + " AND vr.row_id IN (%s)" % (
                    ",".join("?" * len(chunk)))
                for row in conn.execute(q, [version_id, *chunk]):
                    d = self._row_dict(row)
                    out[d["row_id"]] = d
            return out
        finally:
            conn.close()

    def get_group_rows(self, version_id: int, group: str) -> List[dict]:
        """Full group rows including readable body, ordered by row_id."""
        self._ensure()
        conn = self._connect()
        try:
            self._require_version(conn, version_id)
            rows = conn.execute(
                self._ROW_SELECT + " AND vr.group_name=? ORDER BY vr.row_id",
                (version_id, group)).fetchall()
            return [self._row_dict(r) for r in rows]
        finally:
            conn.close()

    def version_groups(self, version_id: int) -> List[str]:
        self._ensure()
        conn = self._connect()
        try:
            self._require_version(conn, version_id)
            rows = conn.execute(
                "SELECT DISTINCT group_name FROM version_rows "
                "WHERE version_id=? ORDER BY group_name", (version_id,)
            ).fetchall()
            return [r["group_name"] for r in rows]
        finally:
            conn.close()

    def group_counts(self, version_id: int) -> Dict[str, int]:
        """Number of active rows per group in a fixed version."""
        self._ensure()
        conn = self._connect()
        try:
            self._require_version(conn, version_id)
            rows = conn.execute(
                "SELECT group_name, COUNT(*) AS n FROM version_rows "
                "WHERE version_id=? GROUP BY group_name ORDER BY group_name",
                (version_id,)).fetchall()
            return {r["group_name"]: int(r["n"]) for r in rows}
        finally:
            conn.close()

    # -- provenance / verification --------------------------------------
    def current_row(self, row_id: str) -> Optional[dict]:
        self._ensure()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT row_id, group_name AS \"group\", weight, rev, active, "
                "shard, body_sha256 FROM rows WHERE row_id=?", (row_id,)
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["active"] = bool(d["active"])
            return d
        finally:
            conn.close()

    def shard_map(self, version_id: Optional[int] = None) -> Dict[str, str]:
        """Current physical row -> shard map (optionally limited to a version)."""
        self._ensure()
        conn = self._connect()
        try:
            if version_id is None:
                rows = conn.execute(
                    "SELECT row_id, shard FROM rows").fetchall()
            else:
                self._require_version(conn, version_id)
                rows = conn.execute(
                    "SELECT vr.row_id, r.shard FROM version_rows vr "
                    "JOIN rows r ON r.row_id=vr.row_id WHERE vr.version_id=?",
                    (version_id,)).fetchall()
            return {r["row_id"]: r["shard"] for r in rows}
        finally:
            conn.close()

    def shards(self) -> List[str]:
        self._ensure()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT shard FROM rows ORDER BY shard").fetchall()
            return [r["shard"] for r in rows]
        finally:
            conn.close()

    def verify_bodies(self, version_id: int) -> int:
        """Raise if any snapshot row lacks a readable, matching body."""
        self._ensure()
        conn = self._connect()
        try:
            self._require_version(conn, version_id)
            total = conn.execute(
                "SELECT COUNT(*) AS n FROM version_rows WHERE version_id=?",
                (version_id,)).fetchone()["n"]
            bad = conn.execute(
                "SELECT COUNT(*) AS n FROM version_rows vr "
                "LEFT JOIN bodies b ON b.sha256=vr.body_sha256 "
                "WHERE vr.version_id=? AND b.sha256 IS NULL",
                (version_id,)).fetchone()["n"]
            if bad:
                raise SourceError(
                    f"{bad} rows in version {version_id} have no body")
            # Confirm digest matches stored bytes.
            for row in conn.execute(
                    "SELECT vr.row_id, vr.body_sha256, b.data FROM version_rows vr "
                    "JOIN bodies b ON b.sha256=vr.body_sha256 "
                    "WHERE vr.version_id=?", (version_id,)):
                import hashlib
                if hashlib.sha256(bytes(row["data"])).hexdigest() != \
                        row["body_sha256"]:
                    raise SourceError(
                        f"body digest mismatch for {row['row_id']!r}")
            return total
        finally:
            conn.close()

    def diff_versions(self, old_version: int, new_version: int) -> dict:
        """Structural change list between two immutable versions."""
        self._ensure()
        conn = self._connect()
        try:
            self._require_version(conn, old_version)
            self._require_version(conn, new_version)
            def fetch(v):
                return {r["row_id"]: {
                    "group": r["group_name"], "weight": r["weight"],
                    "body_sha256": r["body_sha256"], "rev": r["rev"],
                } for r in conn.execute(
                    "SELECT row_id, group_name, weight, body_sha256, rev "
                    "FROM version_rows WHERE version_id=?", (v,))}
            old = fetch(old_version)
            new = fetch(new_version)
        finally:
            conn.close()
        added = sorted(set(new) - set(old))
        removed = sorted(set(old) - set(new))
        changed = []
        for rid in sorted(set(old) & set(new)):
            if old[rid] != new[rid]:
                changed.append({"row_id": rid, "old": old[rid],
                                "new": new[rid]})
        return {
            "old_version": old_version,
            "new_version": new_version,
            "added": added,
            "removed": removed,
            "changed": changed,
            "changed_row_ids": sorted(set(added) | set(removed)
                                      | {c["row_id"] for c in changed}),
            "affected_groups": sorted(
                {new[r]["group"] for r in added}
                | {old[r]["group"] for r in removed}
                | {c["old"]["group"] for c in changed}
                | {c["new"]["group"] for c in changed}),
        }

    def batch_log(self) -> List[dict]:
        self._ensure()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT batch_id, version_id, created_at, note, result_json "
                "FROM batch_log ORDER BY batch_id").fetchall()
            return [{"batch_id": r["batch_id"],
                     "version_id": r["version_id"],
                     "created_at": r["created_at"], "note": r["note"],
                     "result": json.loads(r["result_json"])} for r in rows]
        finally:
            conn.close()

    def dump_version(self, version_id: int, out_path: str) -> int:
        """Write a full version (including bodies) as JSONL, for audit."""
        count = 0
        with open(out_path, "w", encoding="utf-8") as fh:
            for row in self.open_reader(version_id):
                fh.write(json.dumps(row, ensure_ascii=False,
                                    sort_keys=True) + "\n")
                count += 1
        return count


def encode_cursor(version_id: int, after_row_id: str) -> str:
    raw = _canonical_json({"v": version_id, "after": after_row_id})
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


def decode_cursor(cursor: str) -> Tuple[int, str]:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
        obj = json.loads(raw.decode("utf-8"))
        return int(obj["v"]), str(obj["after"])
    except Exception as exc:  # noqa: BLE001 - normalise cursor errors
        raise VersionError("invalid pagination cursor") from exc


class VersionReader:
    """Sequential pagination over one immutable source version."""

    def __init__(self, service: SourceService, version_id: int,
                 page_size: int = 1000):
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        service.get_version(version_id)  # validates
        self.service = service
        self.version_id = version_id
        self.page_size = page_size
        self._cursor: Optional[str] = None
        self._done = False

    def page(self, cursor: Optional[str] = None) -> dict:
        if cursor is None:
            if self._done:
                return {"version": self.version_id, "rows": [],
                        "next_cursor": None, "done": True}
            after = None
        else:
            v, after = decode_cursor(cursor)
            if v != self.version_id:
                raise VersionError(
                    f"cursor belongs to version {v}, reader is {self.version_id}")
        result = self.service.read_page(self.version_id, after,
                                        self.page_size)
        self._cursor = result["next_cursor"]
        self._done = result["done"]
        return result

    def __iter__(self) -> Iterator[dict]:
        cursor = None
        while True:
            result = self.page(cursor)
            yield from result["rows"]
            if result["done"]:
                return
            cursor = result["next_cursor"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_ops(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    data = json.loads(text)
    if isinstance(data, dict) and "ops" in data:
        return data["ops"], data.get("note")
    if isinstance(data, list):
        return data, None
    raise SystemExit("ops file must be a JSON list or {ops:[...],note:...}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wrsource.source",
        description="Revisionable weighted record source (SQLite).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("--db", required=True)

    p = sub.add_parser("batch")
    p.add_argument("--db", required=True)
    p.add_argument("--ops", required=True)
    p.add_argument("--note")

    p = sub.add_parser("versions")
    p.add_argument("--db", required=True)

    p = sub.add_parser("page")
    p.add_argument("--db", required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--cursor")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--out")

    p = sub.add_parser("row")
    p.add_argument("--db", required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--row-id", required=True)

    p = sub.add_parser("diff")
    p.add_argument("--db", required=True)
    p.add_argument("--old", type=int, required=True)
    p.add_argument("--new", type=int, required=True)

    p = sub.add_parser("verify-bodies")
    p.add_argument("--db", required=True)
    p.add_argument("--version", type=int, required=True)

    p = sub.add_parser("dump")
    p.add_argument("--db", required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    svc = SourceService(args.db)

    if args.cmd == "init":
        svc.init()
        print(json.dumps(svc.list_versions(), indent=2))
    elif args.cmd == "batch":
        ops, note = _load_ops(args.ops)
        res = svc.apply_batch(ops, note=args.note or note)
        print(json.dumps(res.as_dict(), indent=2))
    elif args.cmd == "versions":
        print(json.dumps(svc.list_versions(), indent=2))
    elif args.cmd == "page":
        rd = VersionReader(svc, args.version, args.limit)
        res = rd.page(args.cursor)
        out = json.dumps(res, ensure_ascii=False, indent=2)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(out)
        else:
            print(out)
    elif args.cmd == "row":
        row = svc.get_row(args.version, args.row_id)
        print(json.dumps(row, ensure_ascii=False, indent=2))
    elif args.cmd == "diff":
        print(json.dumps(svc.diff_versions(args.old, args.new), indent=2))
    elif args.cmd == "verify-bodies":
        n = svc.verify_bodies(args.version)
        print(json.dumps({"version": args.version, "readable_rows": n}))
    elif args.cmd == "dump":
        n = svc.dump_version(args.version, args.out)
        print(json.dumps({"version": args.version, "rows": n,
                          "out": args.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
