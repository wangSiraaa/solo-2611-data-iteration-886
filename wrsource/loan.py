"""Loan-window quarantine timelines driven by evidence revisions.

A borrower holds the *actual rows* of a signed package for a continuous window
of evidence revisions ``[start_rev, end_rev]``.  For one training plan the
quarantine state of every loaned row is recomputed at each revision from the
same immutable inputs:

* the exposure set is frozen to training batches that had already happened
  before the window (``cutoff_batch``) -- later training batches do not move a
  loan retroactively;
* the same-origin relation is the closure of evidence still valid at that
  revision, so retracted-body nodes still connect and parallel edges keep the
  pair connected until the *last* supporting edge is revoked.

Only real state transitions are reported.  Adding a second independent edge
while the row is already isolated, or revoking one of two supporting edges,
changes nothing and produces no event -- exactly the "add a / add b / revoke a
/ revoke b" sequence in the task, which yields one enter at add-a and one exit
at revoke-b.

The same driver backs both entry points: staff querying a single loan and an
external investigation over many loans share one revision sweep and one
snapshot per revision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .evidence import EvidenceLedger
from .graph import GraphSnapshot
from .package import _exposure_flags, exclusion_reasons, selected_row_ids
from .training import TrainingLedger


@dataclass(frozen=True)
class LoanedPackage:
    """A package physically on loan; only its signed row identities matter."""

    loan_id: str
    package_id: str
    row_ids: List[str]


@dataclass
class QuarantineEvent:
    revision: int
    action: str            # "enter" or "exit"
    row_id: str
    detail: dict


@dataclass
class TimelineResult:
    loan_id: str
    package_id: str
    plan: str
    window_start_revision: int
    window_end_revision: int
    cutoff_batch: int
    initial_quarantined: List[str]
    initial_detail: Dict[str, dict]
    events: List[QuarantineEvent] = field(default_factory=list)

    @property
    def final_quarantined(self) -> List[str]:
        state = set(self.initial_quarantined)
        for ev in self.events:
            if ev.action == "enter":
                state.add(ev.row_id)
            else:
                state.discard(ev.row_id)
        return sorted(state)

    def as_dict(self) -> dict:
        return {
            "loan_id": self.loan_id,
            "package_id": self.package_id,
            "plan": self.plan,
            "window_start_revision": self.window_start_revision,
            "window_end_revision": self.window_end_revision,
            "cutoff_batch": self.cutoff_batch,
            "initial_quarantined": list(self.initial_quarantined),
            "initial_detail": self.initial_detail,
            "events": [
                {"revision": e.revision, "action": e.action,
                 "row_id": e.row_id, "detail": e.detail}
                for e in self.events],
            "final_quarantined": self.final_quarantined,
        }


def _row_state(row_id: str, exposed_flags: Dict[str, bool],
               snap: GraphSnapshot) -> tuple:
    reasons = exclusion_reasons(row_id, exposed_flags, snap)
    return (bool(reasons), reasons)


def _exit_detail(evidence: EvidenceLedger, revision: int, row_id: str,
                 exposed_flags: Dict[str, bool], snap: GraphSnapshot) -> dict:
    op = next((o for o in evidence.op_log() if o["revision"] == revision),
              None)
    return {
        "row_id": row_id,
        "released_by_op": op,
        "explanation": ("after this revision no row identity exposed under "
                        "the plan shares a same-origin component with the "
                        "loaned row at the current evidence revision"),
        "exposed_rows_checked": sorted(exposed_flags),
    }


def loan_timeline(loan: LoanedPackage, plan: str,
                  start_rev: int, end_rev: int,
                  training: TrainingLedger, evidence: EvidenceLedger,
                  cutoff_batch: int,
                  *, snapshot_cache: Optional[Dict[int, GraphSnapshot]] = None
                  ) -> TimelineResult:
    """State at ``start_rev`` plus every real enter/exit up to ``end_rev``."""
    if start_rev < 0 or end_rev < start_rev:
        raise ValueError("require 0 <= start_rev <= end_rev")
    if plan not in training.list_plans():
        raise ValueError(f"unknown plan: {plan!r}")
    exposed_flags = _exposure_flags(training, plan, cutoff_batch)
    cache = snapshot_cache if snapshot_cache is not None else {}

    def snap_at(rev: int) -> GraphSnapshot:
        if rev not in cache:
            cache[rev] = evidence.snapshot(rev)
        return cache[rev]

    rows = list(dict.fromkeys(loan.row_ids))
    initial_detail: Dict[str, dict] = {}
    state: Dict[str, bool] = {}
    first = snap_at(start_rev)
    for rid in rows:
        quarantined, reasons = _row_state(rid, exposed_flags, first)
        state[rid] = quarantined
        initial_detail[rid] = {"quarantined": quarantined,
                               "reasons": reasons}

    result = TimelineResult(
        loan_id=loan.loan_id, package_id=loan.package_id, plan=plan,
        window_start_revision=start_rev, window_end_revision=end_rev,
        cutoff_batch=cutoff_batch,
        initial_quarantined=sorted(r for r, q in state.items() if q),
        initial_detail=initial_detail)

    for rev in range(start_rev + 1, end_rev + 1):
        snap = snap_at(rev)
        for rid in rows:
            quarantined, reasons = _row_state(rid, exposed_flags, snap)
            if quarantined and not state[rid]:
                result.events.append(QuarantineEvent(
                    rev, "enter", rid,
                    {"row_id": rid, "reasons": reasons}))
            elif not quarantined and state[rid]:
                result.events.append(QuarantineEvent(
                    rev, "exit", rid,
                    _exit_detail(evidence, rev, rid, exposed_flags, snap)))
            state[rid] = quarantined
    return result


def investigate_loans(loans: Sequence[LoanedPackage], plan: str,
                      start_rev: int, end_rev: int,
                      training: TrainingLedger, evidence: EvidenceLedger,
                      cutoff_batch: int) -> Dict[str, TimelineResult]:
    """Bulk external investigation: one shared revision sweep for all loans."""
    cache: Dict[int, GraphSnapshot] = {}
    return {
        loan.loan_id: loan_timeline(
            loan, plan, start_rev, end_rev, training, evidence, cutoff_batch,
            snapshot_cache=cache)
        for loan in loans
    }


def loaned_package_from_manifest(loan_id: str, manifest: dict
                                 ) -> LoanedPackage:
    """Convenience: the loaned identity list is the signed selection itself."""
    return LoanedPackage(loan_id, manifest["package_id"],
                         selected_row_ids(manifest))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    from .package import PackageRegistry

    parser = argparse.ArgumentParser(
        prog="python -m wrsource.loan",
        description=("Quarantine timelines for loaned packages over an "
                     "evidence revision window."))
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--training-db", required=True)
        p.add_argument("--evidence-db", required=True)
        p.add_argument("--plan", required=True)
        p.add_argument("--start-rev", type=int, required=True)
        p.add_argument("--end-rev", type=int, required=True)
        p.add_argument("--cutoff-batch", type=int, required=True,
                       help="last training batch already emitted pre-window")

    p = sub.add_parser("timeline")
    common(p)
    p.add_argument("--registry-db", required=True)
    p.add_argument("--package-id", required=True)
    p.add_argument("--loan-id", default="adhoc")

    p = sub.add_parser("investigate")
    common(p)
    p.add_argument("--loans-json", required=True,
                   help=('[{"loan_id":..,"package_id":..}] referencing '
                         'signed packages in --registry-db'))
    p.add_argument("--registry-db", required=True)

    args = parser.parse_args(argv)
    training = TrainingLedger(args.training_db)
    evidence = EvidenceLedger(args.evidence_db)
    registry = PackageRegistry(args.registry_db)

    if args.cmd == "timeline":
        manifest = registry.open(args.package_id)
        loan = loaned_package_from_manifest(args.loan_id, manifest)
        result = loan_timeline(loan, args.plan, args.start_rev,
                               args.end_rev, training, evidence,
                               args.cutoff_batch)
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2,
                         sort_keys=True))
    else:
        loans = []
        with open(args.loans_json, encoding="utf-8") as fh:
            for item in json.load(fh):
                manifest = registry.open(item["package_id"])
                loans.append(loaned_package_from_manifest(
                    item["loan_id"], manifest))
        results = investigate_loans(loans, args.plan, args.start_rev,
                                    args.end_rev, training, evidence,
                                    args.cutoff_batch)
        print(json.dumps({k: v.as_dict() for k, v in results.items()},
                         ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
