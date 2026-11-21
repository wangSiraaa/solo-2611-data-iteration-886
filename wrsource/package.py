"""Evaluation package sampler with same-origin quarantine.

An evaluation package is bound to an immutable triple:

    (training plan, source version, evidence revision)

and one lottery seed plus per-group quotas.  Candidates are the active rows of
the pinned source version; a candidate is *excluded* when, in the pinned
evidence snapshot, its connected component contains any row identity ever
exposed under the training plan (retracted training records still count -- the
underlying original was seen).  The survivors keep the original exact weighted
ordering (``wrsource.sampling``) and fill the group quota from the true next
rank, so backfill is never a resample of the old draw.

Signed packages are immutable.  Rechecking one against newer evidence never
mutates it: the report lists the actual signed rows that must now be isolated
with witness paths, and a separate reissue draws a fresh clean package that
links back to the old one while the old judgment remains on record.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from functools import cmp_to_key
from typing import Dict, List, Optional, Sequence, Tuple

from .evidence import EvidenceLedger
from .graph import GraphSnapshot, OriginPath
from .sampling import (Candidate, boundary_certificate, compare_rank,
                       make_candidate)
from .training import TrainingLedger
from .source import SourceService

MANIFEST_FORMAT = "wreval/1"
MANIFEST_FILE = "manifest.json"
SAMPLES_FILE = "samples.jsonl"


class PackageError(Exception):
    pass


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def _exposure_flags(ledger: TrainingLedger, plan: str,
                    as_of_batch: Optional[int]) -> Dict[str, bool]:
    """Map every exposed row id to its current retracted flag."""
    all_rows = ledger.exposed_rows(plan, include_retracted=True,
                                   as_of_batch=as_of_batch)
    active = set(ledger.exposed_rows(plan, include_retracted=False,
                                     as_of_batch=as_of_batch))
    return {rid: rid not in active for rid in all_rows}


def exclusion_reasons(candidate: str, exposed_flags: Dict[str, bool],
                      snap: GraphSnapshot) -> List[dict]:
    """All exposure witnesses that disqualify ``candidate`` at a snapshot."""
    reasons: List[dict] = []
    for exposed in snap.connected_exposures(candidate,
                                            tuple(exposed_flags)):
        if exposed == candidate and candidate not in exposed_flags:
            continue
        path: Optional[OriginPath]
        path = snap.find_path(exposed, candidate)
        if path is None and exposed == candidate:
            path = OriginPath([candidate], [])
        if path is None:
            continue
        reasons.append({
            "exposed_row_id": exposed,
            "training_record_retracted": bool(exposed_flags[exposed]),
            "path": path.as_dict(),
        })
    reasons.sort(key=lambda r: r["exposed_row_id"])
    return reasons


def explain_eligibility(row_id: str, exposed_flags: Dict[str, bool],
                        snap: GraphSnapshot) -> dict:
    """Why one row is still eligible (or not), reproducible from stored pins."""
    reasons = exclusion_reasons(row_id, exposed_flags, snap)
    if reasons:
        return {"row_id": row_id, "eligible": False, "reasons": reasons}
    return {
        "row_id": row_id,
        "eligible": True,
        "reasons": [],
        "explanation": ("no row identity ever exposed under the pinned "
                        "training plan shares a same-origin component with "
                        "this row at the pinned evidence revision"),
    }


# ---------------------------------------------------------------------------
# Signed package assembly
# ---------------------------------------------------------------------------

@dataclass
class GroupDraw:
    quota: int
    eligible_total: int
    pool: List[Candidate]
    excluded: List[dict] = field(default_factory=list)
    selected: List[Candidate] = field(default_factory=list)
    cut: Optional[dict] = None


def _draw_group(group: str, quota: int, meta_rows: Sequence[dict],
                seed: bytes, exposed_flags: Dict[str, bool],
                snap: GraphSnapshot) -> GroupDraw:
    cands = [make_candidate(r["row_id"], r["group"], r["weight"],
                            r["body_sha256"], seed) for r in meta_rows]
    pool: List[Candidate] = []
    excluded: List[dict] = []
    for cand in cands:
        reasons = exclusion_reasons(cand.row_id, exposed_flags, snap)
        if reasons:
            excluded.append({"row_id": cand.row_id, "reasons": reasons})
        else:
            pool.append(cand)
    ordered = sorted(pool, key=cmp_to_key(compare_rank))
    selected = ordered[:quota] if quota > 0 else []
    cut = None
    if selected and quota < len(ordered):
        cut = boundary_certificate(selected[-1], ordered[quota])
    excluded.sort(key=lambda e: e["row_id"])
    return GroupDraw(quota=quota, eligible_total=len(cands), pool=ordered,
                     excluded=excluded, selected=selected, cut=cut)


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _fingerprint(manifest: dict) -> str:
    body = {k: v for k, v in manifest.items() if k != "fingerprint"}
    return hashlib.sha256(_canonical(body)).hexdigest()


def build_eval_package(service: SourceService, source_version: int,
                       seed: bytes, quotas: Dict[str, int],
                       training: TrainingLedger, plan: str,
                       evidence: EvidenceLedger,
                       evidence_revision: Optional[int] = None,
                       *, cutoff_batch: Optional[int] = None,
                       package_id: Optional[str] = None,
                       parent_package_id: Optional[str] = None,
                       note: Optional[str] = None) -> dict:
    """Draw and sign one evaluation package (no persistence here)."""
    service.get_version(source_version)
    if plan not in training.list_plans():
        raise PackageError(f"unknown training plan: {plan!r}")
    snap = evidence.snapshot(evidence_revision)
    exposed_flags = _exposure_flags(training, plan, cutoff_batch)
    counts = service.group_counts(source_version)
    resolved_quotas: Dict[str, int] = {}
    for g in counts:
        resolved_quotas[g] = int(quotas.get(g, 0))
    for g, q in quotas.items():
        resolved_quotas.setdefault(g, int(q))
    for g, q in resolved_quotas.items():
        if q < 0:
            raise PackageError(f"negative quota for group {g!r}")

    groups_manifest: Dict[str, dict] = {}
    total_selected = 0
    total_excluded = 0
    for g in sorted(resolved_quotas):
        meta = service.group_rows_meta(source_version, g)
        draw = _draw_group(g, resolved_quotas[g], meta, seed,
                           exposed_flags, snap)
        groups_manifest[g] = {
            "quota": draw.quota,
            "eligible_total": draw.eligible_total,
            "excluded_count": len(draw.excluded),
            "selected_count": len(draw.selected),
            "selected": [
                {"rank": i + 1, "row_id": c.row_id, "weight": c.weight,
                 "hash_hex": c.key_hex, "a": str(c.a),
                 "log_screen": c.lkey, "body_sha256": c.body_sha256}
                for i, c in enumerate(draw.selected)],
            "excluded": draw.excluded,
            "cut": draw.cut,
        }
        total_selected += len(draw.selected)
        total_excluded += len(draw.excluded)

    cutoff = (cutoff_batch if cutoff_batch is not None
              else max((b.batch_seq for b in training.batches(plan)),
                       default=0))
    manifest = {
        "format": MANIFEST_FORMAT,
        "package_id": package_id or f"pkg-{uuid.uuid4().hex}",
        "parent_package_id": parent_package_id,
        "created_at": time.time(),
        "note": note,
        "source_version": int(source_version),
        "evidence_revision": int(snap.revision),
        "training_plan": plan,
        "training_cutoff_batch": int(cutoff),
        "seed_hex": seed.hex(),
        "quotas": {g: resolved_quotas[g] for g in sorted(resolved_quotas)},
        "exposure_rows_at_issue": sorted(exposed_flags),
        "totals": {"selected": total_selected,
                   "excluded": total_excluded},
        "groups": groups_manifest,
    }
    manifest["fingerprint"] = _fingerprint(manifest)
    return manifest


def selected_row_ids(manifest: dict) -> List[str]:
    out: List[str] = []
    for g in sorted(manifest["groups"]):
        for rec in manifest["groups"][g]["selected"]:
            out.append(rec["row_id"])
    return out


# ---------------------------------------------------------------------------
# Signed package registry (append-only)
# ---------------------------------------------------------------------------

class PackageRegistry:
    """Immutable store of signed evaluation packages."""

    def __init__(self, path: str):
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
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
                CREATE TABLE IF NOT EXISTS packages(
                    package_id TEXT PRIMARY KEY,
                    parent_package_id TEXT,
                    created_at REAL NOT NULL,
                    fingerprint TEXT NOT NULL,
                    manifest_json TEXT NOT NULL
                );
                COMMIT;
                """
            )
        finally:
            conn.close()

    def _ensure(self):
        if not os.path.exists(self.path):
            self.init()

    def sign(self, manifest: dict) -> str:
        self._ensure()
        expected = _fingerprint(manifest)
        if manifest.get("fingerprint") != expected:
            raise PackageError("manifest fingerprint mismatch; refusing to sign")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "INSERT INTO packages(package_id, parent_package_id, "
                    "created_at, fingerprint, manifest_json) "
                    "VALUES (?,?,?,?,?)",
                    (manifest["package_id"], manifest.get("parent_package_id"),
                     manifest["created_at"], expected,
                     json.dumps(manifest, ensure_ascii=False, sort_keys=True)))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
        return expected

    def issue(self, service: SourceService, source_version: int,
              seed: bytes, quotas: Dict[str, int],
              training: TrainingLedger, plan: str,
              evidence: EvidenceLedger,
              evidence_revision: Optional[int] = None,
              *, cutoff_batch: Optional[int] = None,
              note: Optional[str] = None) -> dict:
        manifest = build_eval_package(
            service, source_version, seed, quotas, training, plan, evidence,
            evidence_revision, cutoff_batch=cutoff_batch, note=note)
        self.sign(manifest)
        return manifest

    def open(self, package_id: str) -> dict:
        self._ensure()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT manifest_json, fingerprint FROM packages "
                "WHERE package_id=?", (package_id,)).fetchone()
            if row is None:
                raise PackageError(f"unknown package: {package_id!r}")
            manifest = json.loads(row["manifest_json"])
            if _fingerprint(manifest) != row["fingerprint"]:
                raise PackageError(
                    f"stored manifest for {package_id!r} fails fingerprint")
            return manifest
        finally:
            conn.close()

    def list_packages(self) -> List[dict]:
        self._ensure()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT package_id, parent_package_id, created_at, "
                "fingerprint FROM packages ORDER BY created_at, package_id"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def export(self, package_id: str, out_dir: str,
               service: SourceService) -> Tuple[str, str]:
        """Write manifest + real bodies JSONL, like the dataset producer."""
        manifest = self.open(package_id)
        version = int(manifest["source_version"])
        ids = selected_row_ids(manifest)
        bodies = service.get_rows(version, ids)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, SAMPLES_FILE), "wb") as fh:
            for gid in sorted(manifest["groups"]):
                for rec in manifest["groups"][gid]["selected"]:
                    row = bodies.get(rec["row_id"])
                    if row is None:
                        raise PackageError(
                            f"row {rec['row_id']!r} unreadable at pinned "
                            f"source version {version}")
                    record = {
                        "package_id": package_id,
                        "group": gid, "rank": rec["rank"],
                        "row_id": rec["row_id"], "weight": rec["weight"],
                        "hash_hex": rec["hash_hex"], "a": rec["a"],
                        "log_screen": rec["log_screen"],
                        "body_sha256": rec["body_sha256"],
                        "body": row["body"],
                        "source_version": version,
                        "seed_hex": manifest["seed_hex"],
                    }
                    fh.write(json.dumps(record, ensure_ascii=False,
                                        sort_keys=True).encode("utf-8"))
                    fh.write(b"\n")
        with open(os.path.join(out_dir, MANIFEST_FILE), "w",
                  encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, sort_keys=True,
                      indent=2)
            fh.write("\n")
        return (os.path.join(out_dir, MANIFEST_FILE),
                os.path.join(out_dir, SAMPLES_FILE))


# ---------------------------------------------------------------------------
# Recheck of an already-signed package against newer evidence
# ---------------------------------------------------------------------------

def recheck_package(manifest: dict, training: TrainingLedger,
                    evidence: EvidenceLedger,
                    *, evidence_revision: Optional[int] = None,
                    cutoff_batch: Optional[int] = None) -> dict:
    """Adjudicate the *actual signed rows* under a (usually newer) revision.

    The signed package is never modified.  Rows now sharing a component with
    any exposure are returned as quarantine items with witness paths; rows
    still clean are listed so the caller sees the full per-row accounting.
    """
    snap = evidence.snapshot(evidence_revision)
    plan = manifest["training_plan"]
    cutoff = (manifest["training_cutoff_batch"] if cutoff_batch is None
              else cutoff_batch)
    exposed_flags = _exposure_flags(training, plan, cutoff)
    quarantine: List[dict] = []
    still_clean: List[str] = []
    for row_id in selected_row_ids(manifest):
        reasons = exclusion_reasons(row_id, exposed_flags, snap)
        if reasons:
            quarantine.append({"row_id": row_id, "reasons": reasons})
        else:
            still_clean.append(row_id)
    quarantine.sort(key=lambda q: q["row_id"])
    return {
        "package_id": manifest["package_id"],
        "signed_evidence_revision": manifest["evidence_revision"],
        "checked_evidence_revision": snap.revision,
        "training_plan": plan,
        "training_cutoff_batch": cutoff,
        "quarantine_required": quarantine,
        "still_clean": still_clean,
        "original_judgment_preserved": True,
    }


def reissue_clean_package(registry: PackageRegistry,
                          service: SourceService,
                          old_package_id: str,
                          training: TrainingLedger,
                          evidence: EvidenceLedger,
                          *, source_version: Optional[int] = None,
                          evidence_revision: Optional[int] = None,
                          cutoff_batch: Optional[int] = None,
                          note: Optional[str] = None) -> Tuple[dict, dict]:
    """Draw a fresh clean package; the old signed package stays untouched.

    Returns ``(new_manifest, reissue_report)`` where the report explains which
    signed rows dropped out (quarantine vs source removal) and which rows
    backfilled their slots.
    """
    old = registry.open(old_package_id)
    seed = bytes.fromhex(old["seed_hex"])
    manifest = build_eval_package(
        service,
        old["source_version"] if source_version is None else source_version,
        seed, old["quotas"], training, old["training_plan"], evidence,
        evidence_revision, cutoff_batch=cutoff_batch,
        parent_package_id=old_package_id,
        note=note or f"clean reissue of {old_package_id}")
    registry.sign(manifest)

    old_rows = set(selected_row_ids(old))
    new_rows = set(selected_row_ids(manifest))
    report = recheck_package(old, training, evidence,
                             evidence_revision=manifest["evidence_revision"],
                             cutoff_batch=manifest["training_cutoff_batch"])
    removed = sorted(old_rows - new_rows)
    added = sorted(new_rows - old_rows)
    present = service.all_rows_meta(manifest["source_version"])
    present_ids = {r["row_id"] for r in present}
    reissue_report = {
        "old_package_id": old_package_id,
        "new_package_id": manifest["package_id"],
        "dropped_rows": [
            {"row_id": rid,
             "fate": ("source_removed" if rid not in present_ids
                      else "quarantined")}
            for rid in removed],
        "backfilled_rows": added,
        "recheck_at_new_revision": report,
        "old_package_immutable": True,
    }
    return manifest, reissue_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_quotas(values: Sequence[str]) -> Dict[str, int]:
    quotas: Dict[str, int] = {}
    for item in values or ():
        group, _, quota = item.partition("=")
        if not group or not quota:
            raise PackageError(f"bad --quota value: {item!r}")
        quotas[group] = int(quota)
    return quotas


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    from .source import SourceService as _Svc

    parser = argparse.ArgumentParser(
        prog="python -m wrsource.package",
        description="Signed evaluation packages with same-origin exclusion.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("issue")
    p.add_argument("--source-db", required=True)
    p.add_argument("--training-db", required=True)
    p.add_argument("--evidence-db", required=True)
    p.add_argument("--registry-db", required=True)
    p.add_argument("--source-version", type=int, required=True)
    p.add_argument("--seed-hex", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--evidence-revision", type=int)
    p.add_argument("--cutoff-batch", type=int)
    p.add_argument("--quota", action="append", default=[])
    p.add_argument("--note")
    p.add_argument("--export-dir")

    p = sub.add_parser("open")
    p.add_argument("--registry-db", required=True)
    p.add_argument("--package-id", required=True)

    p = sub.add_parser("list")
    p.add_argument("--registry-db", required=True)

    p = sub.add_parser("recheck")
    p.add_argument("--registry-db", required=True)
    p.add_argument("--training-db", required=True)
    p.add_argument("--evidence-db", required=True)
    p.add_argument("--package-id", required=True)
    p.add_argument("--evidence-revision", type=int)
    p.add_argument("--cutoff-batch", type=int)

    p = sub.add_parser("reissue")
    p.add_argument("--source-db", required=True)
    p.add_argument("--training-db", required=True)
    p.add_argument("--evidence-db", required=True)
    p.add_argument("--registry-db", required=True)
    p.add_argument("--package-id", required=True)
    p.add_argument("--evidence-revision", type=int)
    p.add_argument("--cutoff-batch", type=int)
    p.add_argument("--note")

    p = sub.add_parser("explain")
    p.add_argument("--training-db", required=True)
    p.add_argument("--evidence-db", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--row-id", required=True)
    p.add_argument("--evidence-revision", type=int)
    p.add_argument("--cutoff-batch", type=int)

    args = parser.parse_args(argv)
    registry = PackageRegistry(args.registry_db) \
        if args.cmd in {"issue", "open", "list", "recheck", "reissue"} \
        else None
    if args.cmd in {"issue", "reissue"}:
        svc = _Svc(args.source_db)
    training = (TrainingLedger(args.training_db)
                if args.cmd in {"issue", "recheck", "reissue", "explain"}
                else None)
    evidence = (EvidenceLedger(args.evidence_db)
                if args.cmd in {"issue", "recheck", "reissue", "explain"}
                else None)

    if args.cmd == "issue":
        quotas = _parse_quotas(args.quota)
        manifest = registry.issue(
            svc, args.source_version, bytes.fromhex(args.seed_hex), quotas,
            training, args.plan, evidence, args.evidence_revision,
            cutoff_batch=args.cutoff_batch, note=args.note)
        out = manifest
        if args.export_dir:
            registry.export(manifest["package_id"], args.export_dir, svc)
    elif args.cmd == "open":
        out = registry.open(args.package_id)
    elif args.cmd == "list":
        out = registry.list_packages()
    elif args.cmd == "recheck":
        out = recheck_package(registry.open(args.package_id), training,
                              evidence,
                              evidence_revision=args.evidence_revision,
                              cutoff_batch=args.cutoff_batch)
    elif args.cmd == "reissue":
        manifest, out = reissue_clean_package(
            registry, svc, args.package_id, training, evidence,
            evidence_revision=args.evidence_revision,
            cutoff_batch=args.cutoff_batch, note=args.note)
    elif args.cmd == "explain":
        snap = evidence.snapshot(args.evidence_revision)
        flags = _exposure_flags(training, args.plan, args.cutoff_batch)
        out = explain_eligibility(args.row_id, flags, snap)
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
