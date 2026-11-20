"""Reproducible weighted dataset producer (full build + incremental refresh).

A *sample* is bound to a complete source version:

    sample = f(source_version, seed, quotas)

The lottery key depends only on ``(seed, row_id)``.  Body edits, group moves
and shard moves therefore never re-key a row, while a weight edit changes the
row's priority (and a group move changes which group's quota it competes for).

Merge rule
----------
Shards are physical only.  Every shard is scanned independently (optionally in
parallel processes) and returns *all* of its candidates.  The parent process
takes the union per group and then applies the exact top-quota rule to the
whole group.  It never concatenates per-shard samples and never keeps a single
global counter, so the result is identical to a single-stream scan regardless
of shard layout or traversal order.

Incremental refresh
-------------------
Given an old dataset directory and a target version, compute the structural
diff.  Every group touched by an add/remove/group-move/weight-edit/body-edit
(or by a quota change) is fully rescanned at the new version.  Untouched
groups keep their old selection; because no row in them changed, this is
identical to reselecting.  A retracted selected row is therefore replaced by
the true next-ranked row, never by resampling the old sample.

Run ``python -m wrsource.dataset --help``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import cmp_to_key
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .sampling import (Candidate, boundary_certificate, compare_rank,
                       log_u, lottery_a)

MANIFEST_FORMAT = "wrsample/1"
SAMPLES_FILE = "samples.jsonl"
MANIFEST_FILE = "manifest.json"
REFRESH_REPORT = "refresh_report.json"


# ---------------------------------------------------------------------------
# Candidate production
# ---------------------------------------------------------------------------

def _compute_chunk(payload):
    """Worker: (seed, [(row_id, group, weight, body_sha256)]) -> candidates.

    Kept at module level so it is picklable by ProcessPoolExecutor.
    """
    seed, rows = payload
    out = []
    for row_id, group, weight, body_sha256 in rows:
        a = lottery_a(seed, row_id)
        out.append((row_id, group, weight, body_sha256, a))
    return out


def _to_candidates(raw) -> List[Candidate]:
    return [Candidate(rid, grp, w, bh, a, log_u(a, w))
            for rid, grp, w, bh, a in raw]


def _rows_meta_to_tuples(meta: Sequence[dict]):
    return [(r["row_id"], r["group"], r["weight"], r["body_sha256"])
            for r in meta]


def _partition_by_shard(meta: Sequence[dict]) -> Dict[str, List[tuple]]:
    shards: Dict[str, List[tuple]] = {}
    for r in meta:
        shards.setdefault(r["shard"], []).append(
            (r["row_id"], r["group"], r["weight"], r["body_sha256"]))
    return shards


def candidates_for_meta(service, seed: bytes, meta: Sequence[dict],
                        parallel: bool = False, workers: int = 0
                        ) -> List[Candidate]:
    """Compute candidates for an explicit row metadata list.

    ``meta`` rows carry the *current* physical shard.  Partitioning is done
    here in the parent from one consistent metadata read, so concurrent shard
    moves cannot cause a row to be missed or duplicated.
    """
    by_shard = _partition_by_shard(meta)
    chunks = [(seed, rows) for rows in by_shard.values() if rows]
    n_workers = workers if workers > 0 else (os.cpu_count() or 1)
    raw: List[tuple] = []
    if parallel and n_workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(chunks))) as ex:
            for part in ex.map(_compute_chunk, chunks):
                raw.extend(part)
    else:
        for chunk in chunks:
            raw.extend(_compute_chunk(chunk))
    return _to_candidates(raw)


def candidates_for_version(service, version: int, seed: bytes,
                           groups: Optional[set] = None, parallel: bool = False,
                           workers: int = 0) -> List[Candidate]:
    meta = service.all_rows_meta(version)
    if groups is not None:
        meta = [r for r in meta if r["group"] in groups]
    return candidates_for_meta(service, seed, meta, parallel, workers)


# ---------------------------------------------------------------------------
# Selection + manifest assembly
# ---------------------------------------------------------------------------

def build_group_result(group: str, quota: int, eligible: int,
                       pool: Sequence[Candidate]) -> dict:
    """Pure selection manifest for one group."""
    if quota < 0:
        raise ValueError(f"negative quota for group {group!r}")
    ordered = sorted(pool, key=cmp_to_key(compare_rank))
    selected = ordered[:quota]
    cut = ordered[quota] if quota < len(ordered) else None
    entry = {
        "quota": quota,
        "eligible": eligible,
        "selected_count": len(selected),
        "selected": [
            {
                "rank": i + 1,
                "row_id": c.row_id,
                "weight": c.weight,
                "hash_hex": c.key_hex,
                "a": str(c.a),
                "log_screen": c.lkey,
                "body_sha256": c.body_sha256,
            }
            for i, c in enumerate(selected)
        ],
    }
    if selected and cut is not None:
        entry["cut"] = boundary_certificate(selected[-1], cut)
    else:
        entry["cut"] = None
    return entry


def _resolve_quotas(version_groups: Iterable[str],
                    quotas: Dict[str, int]) -> Dict[str, int]:
    resolved: Dict[str, int] = {}
    for g in version_groups:
        resolved[g] = int(quotas.get(g, 0))
    for g, q in quotas.items():
        resolved.setdefault(g, int(q))
    for g, q in resolved.items():
        if q < 0:
            raise ValueError(f"negative quota for group {g!r}")
    return resolved


def assemble_manifest(service, version: int, seed: bytes,
                      quotas: Dict[str, int],
                      entries: Dict[str, dict],
                      counts: Dict[str, int],
                      samples_bytes: bytes) -> dict:
    total_selected = sum(e["selected_count"] for e in entries.values())
    return {
        "format": MANIFEST_FORMAT,
        "source_version": version,
        "seed_hex": seed.hex(),
        "quotas": {g: int(quotas.get(g, 0)) for g in sorted(entries)},
        "totals": {
            "eligible": sum(counts.values()),
            "selected": total_selected,
        },
        "records_file": SAMPLES_FILE,
        "records_sha256": hashlib.sha256(samples_bytes).hexdigest(),
        "groups": entries,
    }


def _selected_records(service, version: int, seed: bytes,
                      group_entries: Dict[str, dict]) -> List[dict]:
    """Fetch real bodies at ``version`` for exactly the selected rows."""
    ids: List[str] = []
    for g in sorted(group_entries):
        for rec in group_entries[g]["selected"]:
            ids.append(rec["row_id"])
    bodies = service.get_rows(version, ids)
    records = []
    for g in sorted(group_entries):
        for rec in group_entries[g]["selected"]:
            row = bodies.get(rec["row_id"])
            if row is None:
                raise RuntimeError(
                    f"selected row {rec['row_id']!r} missing from version "
                    f"{version}")
            digest = hashlib.sha256(row["body"].encode("utf-8")).hexdigest()
            if digest != rec["body_sha256"] or digest != row["body_sha256"]:
                raise RuntimeError(
                    f"body digest mismatch for {rec['row_id']!r}")
            records.append({
                "group": g,
                "rank": rec["rank"],
                "row_id": rec["row_id"],
                "weight": rec["weight"],
                "hash_hex": rec["hash_hex"],
                "a": rec["a"],
                "log_screen": rec["log_screen"],
                "body_sha256": digest,
                "body_bytes": len(row["body"].encode("utf-8")),
                "source_version": version,
                "seed_hex": seed.hex(),
                "body": row["body"],
            })
    return records


def _write_jsonl(path: str, records: Sequence[dict]) -> bytes:
    buf = bytearray()
    for rec in records:
        buf.extend(json.dumps(rec, ensure_ascii=False, sort_keys=True)
                   .encode("utf-8"))
        buf.extend(b"\n")
    with open(path, "wb") as fh:
        fh.write(buf)
    return bytes(buf)


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, sort_keys=True, indent=2)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Full build
# ---------------------------------------------------------------------------

def build_dataset(service, version: int, seed: bytes, quotas: Dict[str, int],
                  out_dir: str, parallel: bool = False, workers: int = 0,
                  write: bool = True) -> Tuple[dict, List[dict]]:
    service.get_version(version)
    counts = service.group_counts(version)
    quotas = _resolve_quotas(counts.keys(), quotas)
    needed = {g for g, q in quotas.items() if q > 0}
    cands = candidates_for_version(service, version, seed, groups=needed,
                                   parallel=parallel, workers=workers)
    pools: Dict[str, List[Candidate]] = {}
    for c in cands:
        pools.setdefault(c.group, []).append(c)
    entries = {}
    total = 0
    for g in sorted(set(counts) | set(quotas)):
        entry = build_group_result(g, quotas.get(g, 0), counts.get(g, 0),
                                   pools.get(g, []))
        entries[g] = entry
        total += entry["selected_count"]
    records = _selected_records(service, version, seed, entries)
    records_bytes = b"".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for r in records)
    manifest = assemble_manifest(service, version, seed, quotas, entries,
                                 counts, records_bytes)
    if write:
        os.makedirs(out_dir, exist_ok=True)
        _write_jsonl(os.path.join(out_dir, SAMPLES_FILE), records)
        _write_json(os.path.join(out_dir, MANIFEST_FILE), manifest)
    return manifest, records


# ---------------------------------------------------------------------------
# Incremental refresh
# ---------------------------------------------------------------------------

def load_dataset(dataset_dir: str) -> Tuple[dict, List[dict]]:
    with open(os.path.join(dataset_dir, MANIFEST_FILE), encoding="utf-8") as fh:
        manifest = json.load(fh)
    records = []
    with open(os.path.join(dataset_dir, SAMPLES_FILE), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return manifest, records


def _reuse_group(entry: dict) -> List[Candidate]:
    pool: List[Candidate] = []
    for rec in entry["selected"]:
        a = int(rec["a"])
        pool.append(Candidate(rec["row_id"], "", rec["weight"],
                              rec["body_sha256"], a,
                              log_u(a, rec["weight"])))
    return pool


def refresh_dataset(service, old_dir: str, new_version: Optional[int] = None,
                    quotas: Optional[Dict[str, int]] = None,
                    out_dir: Optional[str] = None, parallel: bool = False,
                    workers: int = 0) -> Tuple[dict, List[dict], dict]:
    old_manifest, old_records = load_dataset(old_dir)
    old_version = int(old_manifest["source_version"])
    if new_version is None:
        new_version = service.latest_version()
    if new_version < old_version:
        raise ValueError("new_version must be >= old version")
    seed = bytes.fromhex(old_manifest["seed_hex"])
    if len(seed) != 32:
        raise ValueError("manifest seed is not 32 bytes")
    if quotas is None:
        quotas = {g: int(q) for g, q in old_manifest["quotas"].items()}

    old_quota = {g: int(q) for g, q in old_manifest["quotas"].items()}
    counts = service.group_counts(new_version)
    quotas = _resolve_quotas(set(counts) | set(quotas), quotas)

    diff = service.diff_versions(old_version, new_version)
    affected = set(diff["affected_groups"])
    for g in set(old_quota) | set(quotas):
        if old_quota.get(g, 0) != quotas.get(g, 0):
            affected.add(g)
    # A group that appears for the first time in the new version is new.
    for g in counts:
        if g not in old_manifest["groups"]:
            affected.add(g)

    needed_scan = {g for g in affected if quotas.get(g, 0) > 0}
    cands = candidates_for_version(service, new_version, seed,
                                   groups=needed_scan, parallel=parallel,
                                   workers=workers)
    scanned: Dict[str, List[Candidate]] = {}
    for c in cands:
        scanned.setdefault(c.group, []).append(c)

    pools: Dict[str, List[Candidate]] = {}
    entries: Dict[str, dict] = {}
    reused_groups: List[str] = []
    rescanned_groups: List[str] = []
    total = 0
    for g in sorted(set(counts) | set(quotas)):
        if g in affected:
            pool = scanned.get(g, [])
            pools[g] = pool
            entry = build_group_result(g, quotas.get(g, 0),
                                       counts.get(g, 0), pool)
            rescanned_groups.append(g)
        elif g in old_manifest["groups"]:
            # Untouched group: identical rows, weights, groups, bodies and
            # quota.  Reusing the frozen entry is exactly equal to rebuilding.
            entry = json.loads(json.dumps(old_manifest["groups"][g]))
            reused_groups.append(g)
            pools[g] = _reuse_group(old_manifest["groups"][g])
        else:  # pragma: no cover - covered by the 'new group' rule above
            pools[g] = []
            entry = build_group_result(g, quotas.get(g, 0), counts.get(g, 0),
                                       [])
            rescanned_groups.append(g)
        # Quota recorded in the new manifest must match the request.
        entry["quota"] = quotas.get(g, 0)
        entries[g] = entry
        total += entry["selected_count"]

    records = _selected_records(service, new_version, seed, entries)
    records_bytes = b"".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for r in records)
    manifest = assemble_manifest(service, new_version, seed, quotas, entries,
                                 counts, records_bytes)

    report = {
        "old_version": old_version,
        "new_version": new_version,
        "seed_hex": seed.hex(),
        "changed_row_ids": diff["changed_row_ids"],
        "added": diff["added"],
        "removed": diff["removed"],
        "changed": [c["row_id"] for c in diff["changed"]],
        "affected_groups": sorted(affected),
        "rescanned_groups": sorted(set(rescanned_groups)),
        "reused_groups": sorted(reused_groups),
        "selected_total": manifest["totals"]["selected"],
    }

    if out_dir is None:
        out_dir = old_dir
    os.makedirs(out_dir, exist_ok=True)
    _write_jsonl(os.path.join(out_dir, SAMPLES_FILE), records)
    _write_json(os.path.join(out_dir, MANIFEST_FILE), manifest)
    _write_json(os.path.join(out_dir, REFRESH_REPORT), report)
    return manifest, records, report


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_dataset(service, dataset_dir: str, require_source: bool = True
                   ) -> dict:
    """Verify a dataset against the source API and a fresh exact rebuild."""
    manifest, records = load_dataset(dataset_dir)
    version = int(manifest["source_version"])
    seed = bytes.fromhex(manifest["seed_hex"])
    quotas = {g: int(q) for g, q in manifest["quotas"].items()}

    raw = b"".join(
        json.dumps(rec, ensure_ascii=False, sort_keys=True).encode("utf-8")
        + b"\n" for rec in records)
    if hashlib.sha256(raw).hexdigest() != manifest["records_sha256"]:
        raise RuntimeError("records_sha256 does not match samples.jsonl")

    # Every delivered row must be re-fetchable from the source version and
    # its body digest must match both the record and the source.
    seen = set()
    if require_source:
        fetched = service.get_rows(version, [r["row_id"] for r in records])
        for rec in records:
            row = fetched.get(rec["row_id"])
            if row is None:
                raise RuntimeError(
                    f"record {rec['row_id']!r} not in source version {version}")
            if row["group"] != rec["group"]:
                raise RuntimeError(f"group mismatch for {rec['row_id']!r}")
            if row["weight"] != rec["weight"]:
                raise RuntimeError(f"weight mismatch for {rec['row_id']!r}")
            digest = hashlib.sha256(row["body"].encode("utf-8")).hexdigest()
            if digest != rec["body_sha256"]:
                raise RuntimeError(f"source body digest mismatch {rec['row_id']!r}")
            if digest != rec["body_sha256"]:
                raise RuntimeError(f"record digest mismatch {rec['row_id']!r}")
            if service_row_key(seed, rec) != rec["hash_hex"]:
                raise RuntimeError(f"hash does not re-key {rec['row_id']!r}")
            seen.add(rec["row_id"])

    # Exact rebuild comparison: identity, order and real bodies.
    rebuild_manifest, rebuild_records = build_dataset(
        service, version, seed, quotas, dataset_dir, write=False)
    if [r["row_id"] for r in records] != [r["row_id"] for r in rebuild_records]:
        raise RuntimeError("sample identity/order differs from exact rebuild")
    for got, want in zip(records, rebuild_records):
        if got["body"] != want["body"]:
            raise RuntimeError(f"body differs for {got['row_id']!r}")
        if got["group"] != want["group"] or got["rank"] != want["rank"]:
            raise RuntimeError(f"placement differs for {got['row_id']!r}")
    if manifest["groups"] != rebuild_manifest["groups"]:
        raise RuntimeError("manifest groups differ from exact rebuild")
    return {
        "ok": True,
        "source_version": version,
        "seed_hex": manifest["seed_hex"],
        "records": len(records),
        "groups": len(manifest["groups"]),
        "records_sha256": manifest["records_sha256"],
    }


def service_row_key(seed: bytes, rec: dict) -> str:
    return format(lottery_a(seed, rec["row_id"]) - 1, "064x")


def compare_datasets(a_dir: str, b_dir: str) -> dict:
    """Byte-level comparison of two dataset directories (refresh vs full)."""
    ma, ra = load_dataset(a_dir)
    mb, rb = load_dataset(b_dir)
    problems = []
    if ma != mb:
        problems.append("manifest differs")
    if ra != rb:
        problems.append("records differ")
    return {"equal": not problems, "problems": problems,
            "records_a": len(ra), "records_b": len(rb)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_quotas(args) -> Dict[str, int]:
    quotas: Dict[str, int] = {}
    if args.quota_json:
        with open(args.quota_json, encoding="utf-8") as fh:
            quotas.update({k: int(v) for k, v in json.load(fh).items()})
    for item in args.quota or []:
        group, _, value = item.partition("=")
        if not group or not value:
            raise SystemExit(f"bad --quota {item!r}; expected GROUP=N")
        quotas[group] = int(value)
    return quotas


def _parse_seed(text: str) -> bytes:
    seed = bytes.fromhex(text)
    if len(seed) != 32:
        raise SystemExit("seed must be 32 bytes (64 hex characters)")
    return seed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wrsource.dataset",
        description="Reproducible weighted dataset producer.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_quota(p):
        p.add_argument("--quota", action="append",
                       help="GROUP=N (repeatable)")
        p.add_argument("--quota-json", help="JSON file mapping group to quota")

    p = sub.add_parser("build")
    p.add_argument("--db", required=True)
    p.add_argument("--version", type=int)
    p.add_argument("--seed-hex", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--workers", type=int, default=0)
    add_quota(p)

    p = sub.add_parser("refresh")
    p.add_argument("--db", required=True)
    p.add_argument("--old", required=True)
    p.add_argument("--new-version", type=int)
    p.add_argument("--out")
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--workers", type=int, default=0)
    add_quota(p)

    p = sub.add_parser("verify")
    p.add_argument("--db", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--no-source", action="store_true")

    p = sub.add_parser("compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)

    p = sub.add_parser("manifest")
    p.add_argument("--dataset", required=True)

    args = parser.parse_args(argv)

    if args.cmd in ("build", "refresh"):
        from .source import SourceService
        svc = SourceService(args.db)
        svc._ensure()
        if args.cmd == "build":
            version = args.version
            if version is None:
                version = svc.latest_version()
            quotas = _parse_quotas(args)
            manifest, records = build_dataset(
                svc, version, _parse_seed(args.seed_hex), quotas, args.out,
                parallel=args.parallel, workers=args.workers)
            print(json.dumps({
                "source_version": version,
                "selected": manifest["totals"]["selected"],
                "groups": len(manifest["groups"]),
                "out": args.out,
            }, indent=2))
        else:
            quotas = _parse_quotas(args)
            quotas_arg = quotas if quotas else None
            manifest, records, report = refresh_dataset(
                svc, args.old, new_version=args.new_version,
                quotas=quotas_arg, out_dir=args.out,
                parallel=args.parallel, workers=args.workers)
            print(json.dumps({
                "source_version": manifest["source_version"],
                "selected": manifest["totals"]["selected"],
                "affected_groups": report["affected_groups"],
                "reused_groups": report["reused_groups"],
            }, indent=2))
    elif args.cmd == "verify":
        from .source import SourceService
        svc = SourceService(args.db)
        svc._ensure()
        print(json.dumps(verify_dataset(
            svc, args.dataset, require_source=not args.no_source), indent=2))
    elif args.cmd == "compare":
        print(json.dumps(compare_datasets(args.a, args.b), indent=2))
    elif args.cmd == "manifest":
        with open(os.path.join(args.dataset, MANIFEST_FILE),
                  encoding="utf-8") as fh:
            print(fh.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
