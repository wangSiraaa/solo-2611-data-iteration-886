#!/usr/bin/env python3
"""End-to-end demonstration of the revisionable source + weighted sampler.

Run from the repository root:

    python3 demo/run_demo.py                # small, fast
    python3 demo/run_demo.py --large        # 100k rows, 100 groups, quota 1000

It creates a fresh working directory under ``demo/_work`` (or ``--work``),
writes a real SQLite source, builds a full sample, applies a mixed revision
batch, incrementally refreshes, verifies everything, and prints a summary plus
the CLI commands a user can replay.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from wrsource.dataset import (build_dataset, compare_datasets, load_dataset,
                              refresh_dataset, verify_dataset)
from wrsource.sampling import make_candidate, select_top
from wrsource.source import SourceService


def banner(text):
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


def write_ops(path, ops):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"ops": ops, "note": os.path.basename(path)}, fh,
                  ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="store_true")
    ap.add_argument("--work", default=os.path.join(HERE, "_work"))
    ap.add_argument("--parallel", action="store_true")
    args = ap.parse_args()

    if os.path.isdir(args.work):
        shutil.rmtree(args.work)
    os.makedirs(args.work)

    n_rows = 100_000 if args.large else 120
    n_groups = 100 if args.large else 5
    quota = 1000 if args.large else 6
    shards = 8 if args.large else 3
    seed = bytes(range(32))
    db = os.path.join(args.work, "source.sqlite")

    banner(f"1. Ingest {n_rows} rows into {n_groups} groups across {shards} "
           "shards")
    svc = SourceService(db)
    svc.init()
    rng = random.Random(1234)
    rows = [{
        "op": "add",
        "row_id": f"row-{i:06d}",
        "group": f"g{i % n_groups:03d}",
        "weight": rng.randint(1, 100),
        "body": f"payload::{i}::" + ("é中" if i % 11 == 0 else "x" * 24),
        "shard": f"s{i % shards}",
    } for i in range(n_rows)]
    t0 = time.time()
    for start in range(0, len(rows), 10_000):
        svc.apply_batch(rows[start:start + 10_000], note="ingest")
    v1 = svc.latest_version()
    print(f"ingested in {time.time() - t0:.2f}s, source version = {v1}")
    print("group_counts sample:", dict(list(svc.group_counts(v1).items())[:3]))
    print("shards:", svc.shards())

    quotas = {f"g{i:03d}": quota for i in range(n_groups)}

    banner("2. Full build (parallel shard scan, exact per-group merge)")
    ds1 = os.path.join(args.work, "ds_v1")
    t0 = time.time()
    m1, recs1 = build_dataset(svc, v1, seed, quotas, ds1,
                              parallel=args.parallel, workers=4)
    print(f"built {m1['totals']['selected']} rows in {time.time() - t0:.2f}s")
    print("records_sha256:", m1["records_sha256"])
    print("first group entry keys:", sorted(m1["groups"]["g000"].keys()))
    cut = m1["groups"]["g000"]["cut"]
    print("cut certificate rule:", cut["rule"] if cut
          else "(none - group not oversubscribed)")

    banner("3. Verify against the source API + exact rebuild")
    print(json.dumps(verify_dataset(svc, ds1), indent=2))

    banner("4. Apply a mixed revision batch")
    g0 = svc.get_group_rows(v1, "g000")
    cands = [make_candidate(r["row_id"], r["group"], r["weight"],
                            r["body_sha256"], seed) for r in g0]
    victim = select_top(cands, 1)[0].row_id           # currently rank 1
    mover = select_top([c for c in cands if c.row_id != victim], 1)[0].row_id
    ops = [
        {"op": "retract", "row_id": victim},
        {"op": "set_group", "row_id": mover, "group": "g001"},
        {"op": "set_weight", "row_id": "row-000002", "weight": 100},
        {"op": "set_body", "row_id": "row-000003", "body": "corrected body"},
        {"op": "set_shard", "row_id": "row-000004", "shard": "hot-shard"},
        {"op": "add", "row_id": "row-NEW-1", "group": "g000", "weight": 100,
         "body": "brand new row", "shard": "s0"},
    ]
    write_ops(os.path.join(args.work, "revision_ops.json"), ops)
    res = svc.apply_batch(ops, note="demo revision")
    v2 = res.version
    print("batch result:", json.dumps(res.as_dict(), indent=2))
    print("affected groups from diff:",
          svc.diff_versions(v1, v2)["affected_groups"])

    banner("5. Incremental refresh + equality with a full rebuild")
    ds2_refresh = os.path.join(args.work, "ds_v2_refresh")
    t0 = time.time()
    m2, recs2, report = refresh_dataset(svc, ds1, new_version=v2,
                                        out_dir=ds2_refresh,
                                        parallel=args.parallel, workers=4)
    print(f"refreshed in {time.time() - t0:.2f}s")
    print("refresh report:", json.dumps(report, indent=2))
    ds2_full = os.path.join(args.work, "ds_v2_full")
    build_dataset(svc, v2, seed, quotas, ds2_full,
                  parallel=args.parallel, workers=4)
    cmp = compare_datasets(ds2_refresh, ds2_full)
    print("refresh == full rebuild:", cmp["equal"], cmp["problems"])
    print("verify refreshed:", verify_dataset(svc, ds2_refresh)["ok"])

    banner("6. Old version still readable; shard moves do not re-key rows")
    print("bodies readable at v1:", svc.verify_bodies(v1),
          "| at v2:", svc.verify_bodies(v2))
    r1 = {r["row_id"]: r for r in recs1}
    r2 = {r["row_id"]: r for r in recs2}
    for rid in ("row-000004",):
        if rid in r1 and rid in r2:
            print(f"{rid}: hash unchanged =",
                  r1[rid]["hash_hex"] == r2[rid]["hash_hex"],
                  "| shard moved to hot-shard")

    banner("Replay with the CLI")
    print(f"""
python -m wrsource.source init    --db {db}
python -m wrsource.source batch   --db {db} --ops {os.path.join(args.work, 'revision_ops.json')}
python -m wrsource.source versions --db {db}
python -m wrsource.source page    --db {db} --version {v1} --limit 50
python -m wrsource.source verify-bodies --db {db} --version {v1}
python -m wrsource.dataset build  --db {db} --version {v1} --seed-hex {seed.hex()} \\
        --quota g000=6 --out {ds1}
python -m wrsource.dataset refresh --db {db} --old {ds1} --new-version {v2} \\
        --out {ds2_refresh}
python -m wrsource.dataset verify --db {db} --dataset {ds2_refresh}
""")
    print("work directory:", args.work)


if __name__ == "__main__":
    main()
