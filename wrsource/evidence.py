"""Revisionable same-origin evidence ledger, backed by SQLite.

The source service revisions *row content*; this ledger independently
revisions *named evidence* that two stable row identities originate from the
same original document.  The two histories never share revision numbers.

Storage model
-------------
* ``evidence`` lists every piece of evidence ever admitted with its normalised
  unordered endpoints.  A row is allowed to keep its node identity after its
  body is retracted, so endpoints are free-form row ids and are not foreign
  keys into the source database.
* ``evidence_ops`` is the append-only revision log: one add or revoke per
  revision, each with the op kind, endpoints and human note.  Revision 0 is
  the empty graph.
* A revision's relation is the transitive closure of all edges whose ``add``
  is at or before the revision and which have not been revoked at or before
  it.  Re-adding the same endpoints under a *new* evidence id is a separate
  edge (parallel evidence); revoking one leaves the other holding the pair.

Snapshots are rebuilt on demand and cached per connection-less service
instance; old revisions stay answerable forever because the log is immutable.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .graph import ActiveEdge, GraphSnapshot, OriginPath

SCHEMA_VERSION = 1


class EvidenceError(Exception):
    pass


def _check_id(value, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{what} must be a non-empty string")
    if "\x00" in value:
        raise EvidenceError(f"{what} must not contain NUL")
    return value


def _normalise(a: str, b: str) -> Tuple[str, str]:
    _check_id(a, "endpoint a")
    _check_id(b, "endpoint b")
    if a == b:
        raise EvidenceError("evidence endpoints must be two distinct rows")
    return (a, b) if a <= b else (b, a)


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    endpoint_a: str
    endpoint_b: str
    added_rev: int
    revoked_rev: Optional[int]

    @property
    def active(self) -> bool:
        return self.revoked_rev is None

    def as_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "endpoint_a": self.endpoint_a,
            "endpoint_b": self.endpoint_b,
            "added_rev": self.added_rev,
            "revoked_rev": self.revoked_rev,
            "active": self.active,
        }


class EvidenceLedger:
    """Append-only evidence revision store with arbitrary historical reads."""

    def __init__(self, path: str):
        self.path = path
        self._snapshot_cache: Dict[int, GraphSnapshot] = {}

    # -- connection / schema ---------------------------------------------
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
                CREATE TABLE IF NOT EXISTS evidence(
                    evidence_id TEXT PRIMARY KEY,
                    endpoint_a TEXT NOT NULL,
                    endpoint_b TEXT NOT NULL,
                    added_rev INTEGER NOT NULL,
                    revoked_rev INTEGER
                );
                CREATE INDEX IF NOT EXISTS ix_evidence_pair
                    ON evidence(endpoint_a, endpoint_b);
                CREATE TABLE IF NOT EXISTS evidence_ops(
                    revision INTEGER PRIMARY KEY,
                    op TEXT NOT NULL CHECK(op IN ('add','revoke')),
                    evidence_id TEXT NOT NULL,
                    endpoint_a TEXT NOT NULL,
                    endpoint_b TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    note TEXT
                );
                COMMIT;
                """
            )
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM meta WHERE key='schema'").fetchone()
            if row["n"] == 0:
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema', ?)",
                    (str(SCHEMA_VERSION),))
        finally:
            conn.close()

    def _ensure(self):
        if not os.path.exists(self.path):
            self.init()

    # -- mutations --------------------------------------------------------
    def add_evidence(self, evidence_id: str, a: str, b: str,
                     note: Optional[str] = None) -> int:
        """Admit a named undirected edge; returns the new revision number."""
        self._ensure()
        eid = _check_id(evidence_id, "evidence_id")
        lo, hi = _normalise(a, b)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if conn.execute(
                        "SELECT 1 FROM evidence WHERE evidence_id=?",
                        (eid,)).fetchone() is not None:
                    raise EvidenceError(
                        f"duplicate evidence_id: {eid!r}")
                rev = self._next_revision(conn)
                conn.execute(
                    "INSERT INTO evidence(evidence_id, endpoint_a, "
                    "endpoint_b, added_rev, revoked_rev) VALUES (?,?,?,?,NULL)",
                    (eid, lo, hi, rev))
                conn.execute(
                    "INSERT INTO evidence_ops(revision, op, evidence_id, "
                    "endpoint_a, endpoint_b, created_at, note) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (rev, "add", eid, lo, hi, time.time(), note))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
        self._snapshot_cache.clear()
        return rev

    def revoke_evidence(self, evidence_id: str,
                        note: Optional[str] = None) -> int:
        """Revoke one named edge; parallel edges on the pair are untouched."""
        self._ensure()
        eid = _check_id(evidence_id, "evidence_id")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT evidence_id, endpoint_a, endpoint_b, revoked_rev "
                    "FROM evidence WHERE evidence_id=?", (eid,)).fetchone()
                if row is None:
                    raise EvidenceError(f"unknown evidence_id: {eid!r}")
                if row["revoked_rev"] is not None:
                    raise EvidenceError(
                        f"evidence {eid!r} already revoked at "
                        f"revision {row['revoked_rev']}")
                rev = self._next_revision(conn)
                conn.execute(
                    "UPDATE evidence SET revoked_rev=? WHERE evidence_id=?",
                    (rev, eid))
                conn.execute(
                    "INSERT INTO evidence_ops(revision, op, evidence_id, "
                    "endpoint_a, endpoint_b, created_at, note) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (rev, "revoke", eid, row["endpoint_a"],
                     row["endpoint_b"], time.time(), note))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
        self._snapshot_cache.clear()
        return rev

    @staticmethod
    def _next_revision(conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(revision), 0) AS r FROM evidence_ops"
        ).fetchone()
        return int(row["r"]) + 1

    # -- historical reads -------------------------------------------------
    def latest_revision(self) -> int:
        self._ensure()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COALESCE(MAX(revision), 0) AS r FROM evidence_ops"
            ).fetchone()
            return int(row["r"])
        finally:
            conn.close()

    def _require_revision(self, conn: sqlite3.Connection, revision: int
                          ) -> None:
        if revision < 0:
            raise EvidenceError(f"unknown revision: {revision}")
        row = conn.execute(
            "SELECT COALESCE(MAX(revision), 0) AS r FROM evidence_ops"
        ).fetchone()
        if revision > int(row["r"]):
            raise EvidenceError(f"unknown revision: {revision}")

    def _active_rows(self, conn: sqlite3.Connection, revision: int
                     ) -> List[sqlite3.Row]:
        return conn.execute(
            "SELECT evidence_id, endpoint_a, endpoint_b FROM evidence "
            "WHERE added_rev <= ? AND "
            "(revoked_rev IS NULL OR revoked_rev > ?) "
            "ORDER BY evidence_id", (revision, revision)).fetchall()

    def snapshot(self, revision: Optional[int] = None) -> GraphSnapshot:
        """Immutable closure view at ``revision`` (default: latest)."""
        self._ensure()
        conn = self._connect()
        try:
            if revision is None:
                revision = self.latest_revision()
            cached = self._snapshot_cache.get(revision)
            if cached is not None:
                return cached
            self._require_revision(conn, revision)
            edges = [ActiveEdge(r["evidence_id"], r["endpoint_a"],
                                r["endpoint_b"])
                     for r in self._active_rows(conn, revision)]
            snap = GraphSnapshot(revision, edges)
            self._snapshot_cache[revision] = snap
            return snap
        finally:
            conn.close()

    def same_origin(self, a: str, b: str,
                    revision: Optional[int] = None) -> bool:
        return self.snapshot(revision).same_origin(a, b)

    def origin_path(self, a: str, b: str,
                    revision: Optional[int] = None
                    ) -> Optional[OriginPath]:
        """BFS witness (nodes + evidence ids) at a revision, or None."""
        return self.snapshot(revision).find_path(a, b)

    def connected_exposures(self, node: str, exposed: Sequence[str],
                            revision: Optional[int] = None) -> List[str]:
        return self.snapshot(revision).connected_exposures(node, exposed)

    # -- audit views ------------------------------------------------------
    def get_evidence(self, evidence_id: str) -> Optional[EvidenceRecord]:
        self._ensure()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT evidence_id, endpoint_a, endpoint_b, added_rev, "
                "revoked_rev FROM evidence WHERE evidence_id=?",
                (evidence_id,)).fetchone()
            if row is None:
                return None
            return EvidenceRecord(row["evidence_id"], row["endpoint_a"],
                                  row["endpoint_b"], int(row["added_rev"]),
                                  None if row["revoked_rev"] is None
                                  else int(row["revoked_rev"]))
        finally:
            conn.close()

    def list_evidence(self, include_inactive: bool = True
                      ) -> List[EvidenceRecord]:
        self._ensure()
        sql = ("SELECT evidence_id, endpoint_a, endpoint_b, added_rev, "
               "revoked_rev FROM evidence")
        if not include_inactive:
            sql += " WHERE revoked_rev IS NULL"
        sql += " ORDER BY evidence_id"
        conn = self._connect()
        try:
            return [EvidenceRecord(r["evidence_id"], r["endpoint_a"],
                                   r["endpoint_b"], int(r["added_rev"]),
                                   None if r["revoked_rev"] is None
                                   else int(r["revoked_rev"]))
                    for r in conn.execute(sql)]
        finally:
            conn.close()

    def evidence_between(self, a: str, b: str
                         ) -> List[EvidenceRecord]:
        """Every piece of evidence (active or not) for one unordered pair."""
        lo, hi = _normalise(a, b)
        self._ensure()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT evidence_id, endpoint_a, endpoint_b, added_rev, "
                "revoked_rev FROM evidence WHERE endpoint_a=? AND endpoint_b=? "
                "ORDER BY added_rev, evidence_id", (lo, hi)).fetchall()
            return [EvidenceRecord(r["evidence_id"], r["endpoint_a"],
                                   r["endpoint_b"], int(r["added_rev"]),
                                   None if r["revoked_rev"] is None
                                   else int(r["revoked_rev"]))
                    for r in rows]
        finally:
            conn.close()

    def op_log(self) -> List[dict]:
        self._ensure()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT revision, op, evidence_id, endpoint_a, endpoint_b, "
                "created_at, note FROM evidence_ops ORDER BY revision"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def dump_json(self, out_path: str) -> int:
        payload = {
            "schema": SCHEMA_VERSION,
            "latest_revision": self.latest_revision(),
            "evidence": [e.as_dict() for e in self.list_evidence()],
            "ops": self.op_log(),
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, sort_keys=True,
                      indent=2)
        return len(payload["ops"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m wrsource.evidence",
        description="Revisionable same-origin evidence ledger.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("--db", required=True)

    p = sub.add_parser("add")
    p.add_argument("--db", required=True)
    p.add_argument("--evidence-id", required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--note")

    p = sub.add_parser("revoke")
    p.add_argument("--db", required=True)
    p.add_argument("--evidence-id", required=True)
    p.add_argument("--note")

    p = sub.add_parser("same-origin")
    p.add_argument("--db", required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--revision", type=int)

    p = sub.add_parser("path")
    p.add_argument("--db", required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--revision", type=int)

    p = sub.add_parser("list")
    p.add_argument("--db", required=True)

    p = sub.add_parser("log")
    p.add_argument("--db", required=True)

    p = sub.add_parser("dump")
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    led = EvidenceLedger(args.db)

    if args.cmd == "init":
        led.init()
        print(json.dumps({"db": args.db,
                          "latest_revision": led.latest_revision()}))
    elif args.cmd == "add":
        rev = led.add_evidence(args.evidence_id, args.a, args.b, args.note)
        print(json.dumps({"revision": rev}))
    elif args.cmd == "revoke":
        rev = led.revoke_evidence(args.evidence_id, args.note)
        print(json.dumps({"revision": rev}))
    elif args.cmd == "same-origin":
        rev = args.revision if args.revision is not None \
            else led.latest_revision()
        path = led.origin_path(args.a, args.b, rev)
        print(json.dumps({
            "a": args.a, "b": args.b, "revision": rev,
            "same_origin": led.same_origin(args.a, args.b, rev),
            "witness": None if path is None else path.as_dict(),
        }, ensure_ascii=False, indent=2))
    elif args.cmd == "path":
        rev = args.revision if args.revision is not None \
            else led.latest_revision()
        path = led.origin_path(args.a, args.b, rev)
        print(json.dumps(None if path is None else path.as_dict(),
                         ensure_ascii=False, indent=2))
    elif args.cmd == "list":
        print(json.dumps([e.as_dict() for e in led.list_evidence()],
                         ensure_ascii=False, indent=2))
    elif args.cmd == "log":
        print(json.dumps(led.op_log(), ensure_ascii=False, indent=2))
    elif args.cmd == "dump":
        n = led.dump_json(args.out)
        print(json.dumps({"ops": n, "out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
