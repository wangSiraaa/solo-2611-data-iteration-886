"""Training exposure ledger: which original row identities were ever shown.

The sampler must not reconstruct historical training rosters from *today's*
source.  Exposure is therefore a first-class append-only record: every batch
of raw row identities actually emitted by a training product is registered
under a named training plan, together with the evidence/source revisions that
were current when it went out (metadata only; the row ids are the load-bearing
part).

Two nuances from the task:

* A training record may later be *retracted from training* (mistakenly shipped,
  then withdrawn).  Retraction never deletes history: the row's underlying
  original was seen, so it stays an exposure for every quarantine decision.
* Plans evolve independently.  A later exposure batch only adds rows; queries
  can answer both "everything ever exposed" and "what was known at batch N" so
  a loan-window investigation can ignore exposures that had not happened yet.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

SCHEMA_VERSION = 1


class TrainingError(Exception):
    pass


def _check_name(value: str, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrainingError(f"{what} must be a non-empty string")
    if "\x00" in value:
        raise TrainingError(f"{what} must not contain NUL")
    return value


@dataclass(frozen=True)
class ExposureBatch:
    plan: str
    batch_seq: int
    row_ids: List[str]
    source_version: Optional[int]
    evidence_revision: Optional[int]
    note: Optional[str]
    created_at: float

    def as_dict(self) -> dict:
        return {
            "plan": self.plan,
            "batch_seq": self.batch_seq,
            "row_ids": list(self.row_ids),
            "source_version": self.source_version,
            "evidence_revision": self.evidence_revision,
            "note": self.note,
            "created_at": self.created_at,
        }


@dataclass
class PlanState:
    plan: str
    batches: int
    exposed: List[str] = field(default_factory=list)
    retracted: List[str] = field(default_factory=list)


class TrainingLedger:
    """Append-only per-plan registry of emitted training row identities."""

    def __init__(self, path: str):
        self.path = path

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
                CREATE TABLE IF NOT EXISTS plans(
                    plan TEXT PRIMARY KEY,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batches(
                    plan TEXT NOT NULL REFERENCES plans(plan),
                    batch_seq INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    source_version INTEGER,
                    evidence_revision INTEGER,
                    note TEXT,
                    PRIMARY KEY(plan, batch_seq)
                );
                CREATE TABLE IF NOT EXISTS exposures(
                    plan TEXT NOT NULL,
                    batch_seq INTEGER NOT NULL,
                    row_id TEXT NOT NULL,
                    retracted INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(plan, row_id),
                    FOREIGN KEY(plan, batch_seq)
                        REFERENCES batches(plan, batch_seq)
                );
                CREATE INDEX IF NOT EXISTS ix_exposures_batch
                    ON exposures(plan, batch_seq);
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
    def create_plan(self, plan: str) -> None:
        self._ensure()
        name = _check_name(plan, "plan")
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO plans(plan, created_at) VALUES (?, ?) "
                "ON CONFLICT(plan) DO NOTHING", (name, time.time()))
        finally:
            conn.close()

    def register_batch(self, plan: str, row_ids: Sequence[str],
                       source_version: Optional[int] = None,
                       evidence_revision: Optional[int] = None,
                       note: Optional[str] = None) -> int:
        """Record one batch of row identities actually sent to training.

        Returns the new batch sequence number (1-based within the plan).
        Duplicate row ids inside or across batches are collapsed in the
        exposure set but every batch keeps its own member list.
        """
        self._ensure()
        name = _check_name(plan, "plan")
        ids = [_check_name(r, "row_id") for r in row_ids]
        if not ids:
            raise TrainingError("an exposure batch must contain at least one row")
        unique_ids = list(dict.fromkeys(ids))
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if conn.execute("SELECT 1 FROM plans WHERE plan=?",
                                (name,)).fetchone() is None:
                    raise TrainingError(f"unknown plan: {name!r}")
                row = conn.execute(
                    "SELECT COALESCE(MAX(batch_seq), 0) AS s FROM batches "
                    "WHERE plan=?", (name,)).fetchone()
                seq = int(row["s"]) + 1
                conn.execute(
                    "INSERT INTO batches(plan, batch_seq, created_at, "
                    "source_version, evidence_revision, note) "
                    "VALUES (?,?,?,?,?,?)",
                    (name, seq, time.time(), source_version,
                     evidence_revision, note))
                for rid in unique_ids:
                    # First emission wins for batch attribution; re-emission
                    # of a known row is a no-op but stays inside the batch log.
                    conn.execute(
                        "INSERT INTO exposures(plan, batch_seq, row_id, "
                        "retracted) VALUES (?,?,?,0) "
                        "ON CONFLICT(plan, row_id) DO NOTHING",
                        (name, seq, rid))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
        return seq

    def retract_exposure(self, plan: str, row_id: str,
                         note: Optional[str] = None) -> None:
        """Mark a training record withdrawn. History is preserved."""
        self._ensure()
        name = _check_name(plan, "plan")
        rid = _check_name(row_id, "row_id")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute(
                    "UPDATE exposures SET retracted=1 WHERE plan=? AND "
                    "row_id=?", (name, rid))
                if cur.rowcount == 0:
                    raise TrainingError(
                        f"row {rid!r} was never exposed under plan {name!r}")
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

    # -- queries ----------------------------------------------------------
    def list_plans(self) -> List[str]:
        self._ensure()
        conn = self._connect()
        try:
            return [r["plan"] for r in conn.execute(
                "SELECT plan FROM plans ORDER BY plan")]
        finally:
            conn.close()

    def exposed_rows(self, plan: str, include_retracted: bool = True,
                     as_of_batch: Optional[int] = None) -> List[str]:
        """Row identities ever emitted under a plan.

        ``as_of_batch`` restricts to batches with sequence <= the cutoff, so a
        windowed investigation sees only exposures that had occurred then.
        Retracted rows are included by default: a seen original stays seen.
        """
        self._ensure()
        name = _check_name(plan, "plan")
        sql = ("SELECT DISTINCT row_id FROM exposures WHERE plan=?")
        params: List[object] = [name]
        if as_of_batch is not None:
            if as_of_batch < 0:
                raise TrainingError("as_of_batch must be >= 0")
            sql += " AND batch_seq <= ?"
            params.append(int(as_of_batch))
        if not include_retracted:
            sql += " AND retracted=0"
        sql += " ORDER BY row_id"
        conn = self._connect()
        try:
            return [r["row_id"]
                    for r in conn.execute(sql, params)]
        finally:
            conn.close()

    def is_exposed(self, plan: str, row_id: str,
                   as_of_batch: Optional[int] = None) -> bool:
        return row_id in set(
            self.exposed_rows(plan, as_of_batch=as_of_batch))

    def batches(self, plan: str) -> List[ExposureBatch]:
        self._ensure()
        name = _check_name(plan, "plan")
        conn = self._connect()
        try:
            brows = conn.execute(
                "SELECT plan, batch_seq, created_at, source_version, "
                "evidence_revision, note FROM batches WHERE plan=? "
                "ORDER BY batch_seq", (name,)).fetchall()
            out: List[ExposureBatch] = []
            for b in brows:
                members = [r["row_id"] for r in conn.execute(
                    "SELECT row_id FROM exposures WHERE plan=? AND "
                    "batch_seq=? ORDER BY row_id",
                    (name, b["batch_seq"]))]
                out.append(ExposureBatch(
                    plan=b["plan"], batch_seq=int(b["batch_seq"]),
                    row_ids=members,
                    source_version=None if b["source_version"] is None
                    else int(b["source_version"]),
                    evidence_revision=None if b["evidence_revision"] is None
                    else int(b["evidence_revision"]),
                    note=b["note"], created_at=float(b["created_at"])))
            return out
        finally:
            conn.close()

    def plan_state(self, plan: str) -> PlanState:
        self._ensure()
        name = _check_name(plan, "plan")
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM batches WHERE plan=?",
                (name,)).fetchone()
            if row["n"] == 0 and not self._plan_exists(conn, name):
                raise TrainingError(f"unknown plan: {name!r}")
            exposed = [r["row_id"] for r in conn.execute(
                "SELECT row_id FROM exposures WHERE plan=? AND retracted=0 "
                "ORDER BY row_id", (name,))]
            retracted = [r["row_id"] for r in conn.execute(
                "SELECT row_id FROM exposures WHERE plan=? AND retracted=1 "
                "ORDER BY row_id", (name,))]
            return PlanState(name, int(row["n"]), exposed, retracted)
        finally:
            conn.close()

    @staticmethod
    def _plan_exists(conn: sqlite3.Connection, plan: str) -> bool:
        return conn.execute("SELECT 1 FROM plans WHERE plan=?",
                            (plan,)).fetchone() is not None

    def dump_json(self, out_path: str) -> int:
        payload = {
            "schema": SCHEMA_VERSION,
            "plans": [],
        }
        for plan in self.list_plans():
            state = self.plan_state(plan)
            payload["plans"].append({
                "plan": plan,
                "batches": [b.as_dict() for b in self.batches(plan)],
                "exposed_active": state.exposed,
                "exposed_retracted": state.retracted,
                "exposed_all": self.exposed_rows(plan),
            })
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, sort_keys=True,
                      indent=2)
        return len(payload["plans"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m wrsource.training",
        description="Append-only training exposure ledger.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("--db", required=True)

    p = sub.add_parser("create-plan")
    p.add_argument("--db", required=True)
    p.add_argument("--plan", required=True)

    p = sub.add_parser("register")
    p.add_argument("--db", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--rows", required=True,
                   help="comma-separated row identities actually emitted")
    p.add_argument("--source-version", type=int)
    p.add_argument("--evidence-revision", type=int)
    p.add_argument("--note")

    p = sub.add_parser("retract")
    p.add_argument("--db", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--row-id", required=True)

    p = sub.add_parser("exposed")
    p.add_argument("--db", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--as-of-batch", type=int)
    p.add_argument("--active-only", action="store_true")

    p = sub.add_parser("batches")
    p.add_argument("--db", required=True)
    p.add_argument("--plan", required=True)

    p = sub.add_parser("dump")
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    led = TrainingLedger(args.db)

    if args.cmd == "init":
        led.init()
        print(json.dumps({"db": args.db, "plans": led.list_plans()}))
    elif args.cmd == "create-plan":
        led.create_plan(args.plan)
        print(json.dumps({"plan": args.plan}))
    elif args.cmd == "register":
        rows = [r for r in args.rows.split(",") if r]
        seq = led.register_batch(args.plan, rows, args.source_version,
                                 args.evidence_revision, args.note)
        print(json.dumps({"plan": args.plan, "batch_seq": seq,
                          "row_count": len(rows)}))
    elif args.cmd == "retract":
        led.retract_exposure(args.plan, args.row_id)
        print(json.dumps({"plan": args.plan, "row_id": args.row_id,
                          "history_preserved": True}))
    elif args.cmd == "exposed":
        print(json.dumps(led.exposed_rows(
            args.plan, include_retracted=not args.active_only,
            as_of_batch=args.as_of_batch), indent=2))
    elif args.cmd == "batches":
        print(json.dumps([b.as_dict() for b in led.batches(args.plan)],
                         ensure_ascii=False, indent=2))
    elif args.cmd == "dump":
        n = led.dump_json(args.out)
        print(json.dumps({"plans": n, "out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
