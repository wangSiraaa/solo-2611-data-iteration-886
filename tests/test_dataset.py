import hashlib
import json
import os
import random
import shutil
import tempfile
import unittest

from wrsource.dataset import (compare_datasets, build_dataset, load_dataset,
                              refresh_dataset, verify_dataset)
from wrsource.reference import select_reference
from wrsource.sampling import make_candidate, select_top
from wrsource.source import SourceService


def add(i, group=None, weight=None, shard=None, body=None):
    return {
        "op": "add",
        "row_id": f"r{i:04d}",
        "group": group if group is not None else f"g{i % 5}",
        "weight": weight if weight is not None else (i * 7 % 100) + 1,
        "body": body if body is not None else f"row-{i}-body",
        "shard": shard if shard is not None else f"s{i % 4}",
    }


def rows_of(records):
    return {r["row_id"]: r for r in records}


class DatasetBase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.svc = SourceService(os.path.join(self.dir, "src.sqlite"))
        self.svc.init()
        self.seed = bytes(range(32))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def build(self, version, quotas, out, seed=None, parallel=False,
              workers=0):
        return build_dataset(self.svc, version, seed or self.seed, quotas, out,
                             parallel=parallel, workers=workers)

    def refresh(self, old, new_version=None, quotas=None, out=None,
                parallel=False, workers=0):
        return refresh_dataset(self.svc, old, new_version=new_version,
                               quotas=quotas, out_dir=out,
                               parallel=parallel, workers=workers)

    def path(self, name):
        return os.path.join(self.dir, name)


class DatasetBuildTest(DatasetBase):
    def test_full_build_manifest_and_jsonl(self):
        v = self.svc.apply_batch([add(i) for i in range(40)]).version
        m, recs = self.build(v, {"g0": 3, "g1": 3}, self.path("ds"))
        self.assertTrue(os.path.exists(self.path("ds/samples.jsonl")))
        self.assertTrue(os.path.exists(self.path("ds/manifest.json")))
        self.assertEqual(m["source_version"], v)
        self.assertEqual(m["seed_hex"], self.seed.hex())
        self.assertEqual(len(recs), 6)
        with open(self.path("ds/manifest.json"), encoding="utf-8") as fh:
            on_disk = json.load(fh)
        self.assertEqual(on_disk["records_sha256"], m["records_sha256"])
        with open(self.path("ds/samples.jsonl"), "rb") as fh:
            raw = fh.read()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), m["records_sha256"])
        for rec in recs:
            self.assertEqual(rec["source_version"], v)
            self.assertEqual(rec["seed_hex"], self.seed.hex())
            self.assertEqual(
                hashlib.sha256(rec["body"].encode()).hexdigest(),
                rec["body_sha256"])
            self.assertIn("a", rec)
            self.assertIn("hash_hex", rec)
        # Manifest carries exact comparison material and a cut certificate.
        for g in ("g0", "g1"):
            entry = m["groups"][g]
            self.assertEqual(entry["selected_count"], 3)
            self.assertIsNotNone(entry["cut"])
            self.assertGreater(int(entry["cut"]["left_exact"]),
                               int(entry["cut"]["right_exact"]))

    def test_matches_independent_reference(self):
        rows = [add(i) for i in range(60)]
        v = self.svc.apply_batch(rows).version
        quotas = {"g0": 5, "g1": 0, "g2": 4, "g3": 30, "g4": 2}
        m, recs = self.build(v, quotas, self.path("ds"))
        meta = [{"row_id": r["row_id"], "group": r["group"],
                 "weight": r["weight"]} for r in rows]
        ref = select_reference(self.seed, meta, quotas)
        got = {}
        for rec in recs:
            got.setdefault(rec["group"], []).append(rec["row_id"])
        for g in ref:
            self.assertEqual(got.get(g, []), ref[g], msg=f"group {g}")

    def test_less_than_quota_selects_all(self):
        v = self.svc.apply_batch([add(i, group="g0") for i in range(4)]).version
        m, recs = self.build(v, {"g0": 10}, self.path("ds"))
        self.assertEqual(len(recs), 4)
        self.assertEqual(m["groups"]["g0"]["selected_count"], 4)
        self.assertIsNone(m["groups"]["g0"]["cut"])

    def test_zero_quota_empty(self):
        v = self.svc.apply_batch([add(i, group="g0") for i in range(4)]).version
        m, recs = self.build(v, {"g0": 0}, self.path("ds"))
        self.assertEqual(recs, [])
        self.assertEqual(m["groups"]["g0"]["selected_count"], 0)
        self.assertIsNone(m["groups"]["g0"]["cut"])

    def test_bodies_come_from_requested_version(self):
        v1 = self.svc.apply_batch([add(i, group="g0") for i in range(6)]).version
        v2 = self.svc.apply_batch(
            [{"op": "set_body", "row_id": "r0000", "body": "CORRECTED"}
             ]).version
        m1, recs1 = self.build(v1, {"g0": 6}, self.path("ds1"))
        m2, recs2 = self.build(v2, {"g0": 6}, self.path("ds2"))
        # Same row set, same order, same lottery keys; only body differs.
        self.assertEqual([r["row_id"] for r in recs1],
                         [r["row_id"] for r in recs2])
        r1 = rows_of(recs1)["r0000"]
        r2 = rows_of(recs2)["r0000"]
        self.assertEqual(r1["hash_hex"], r2["hash_hex"])
        self.assertEqual(r1["a"], r2["a"])
        self.assertEqual(r1["body"], "row-0-body")
        self.assertEqual(r2["body"], "CORRECTED")

    def test_weight_change_reorders(self):
        v1 = self.svc.apply_batch([add(i, group="g0") for i in range(20)]).version
        m1, recs1 = self.build(v1, {"g0": 5}, self.path("ds1"))
        # Give a currently-unselected row the maximum weight.
        selected = {r["row_id"] for r in recs1}
        candidate = next(f"r{i:04d}" for i in range(20)
                         if f"r{i:04d}" not in selected)
        v2 = self.svc.apply_batch(
            [{"op": "set_weight", "row_id": candidate, "weight": 100}]).version
        m2, recs2 = self.build(v2, {"g0": 5}, self.path("ds2"))
        # Recompute exactly: the key is unchanged, the weight is 100.
        cands = []
        for r in self.svc.get_group_rows(v2, "g0"):
            cands.append(make_candidate(r["row_id"], r["group"], r["weight"],
                                        r["body_sha256"], self.seed))
        expect = [c.row_id for c in select_top(cands, 5)]
        self.assertEqual([r["row_id"] for r in recs2], expect)

    def test_shard_layout_invariance(self):
        v = self.svc.apply_batch(
            [add(i, shard=f"s{i % 4}") for i in range(80)]).version
        quotas = {"g0": 8, "g1": 8, "g2": 8, "g3": 8, "g4": 8}
        m1, recs1 = self.build(v, quotas, self.path("ds_a"), parallel=False)
        m2, recs2 = self.build(v, quotas, self.path("ds_par"), parallel=True,
                               workers=4)
        self.assertEqual(compare_datasets(self.path("ds_a"),
                                          self.path("ds_par"))["equal"], True)
        # Move every row to one shard, in a fixed version with no content
        # change: selection must be byte-for-byte identical.
        self.svc.apply_batch(
            [{"op": "set_shard", "row_id": f"r{i:04d}", "shard": "one"}
             for i in range(80)])
        m3, recs3 = self.build(v, quotas, self.path("ds_one"),
                               parallel=True, workers=2)
        self.assertEqual(compare_datasets(self.path("ds_a"),
                                          self.path("ds_one"))["equal"], True)

    def test_parallel_multiprocess_equals_single_stream(self):
        v = self.svc.apply_batch(
            [add(i, shard=f"s{i % 6}") for i in range(120)]).version
        quotas = {f"g{i}": 6 for i in range(5)}
        a = self.build(v, quotas, self.path("a"), parallel=False)
        b = self.build(v, quotas, self.path("b"), parallel=True, workers=3)
        self.assertEqual(compare_datasets(self.path("a"), self.path("b"))[
            "equal"], True)


class DatasetRevisionTest(DatasetBase):
    def test_retract_selected_row_replaced_by_true_next(self):
        v1 = self.svc.apply_batch([add(i, group="g0") for i in range(30)]).version
        m1, recs1 = self.build(v1, {"g0": 5}, self.path("ds1"))
        cands = [make_candidate(r["row_id"], r["group"], r["weight"],
                                r["body_sha256"], self.seed)
                 for r in self.svc.get_group_rows(v1, "g0")]
        full = [c.row_id for c in select_top(cands, 30)]
        victim = full[0]           # current rank 1
        expect = full[1:6]         # ranks 2..6
        v2 = self.svc.apply_batch([{"op": "retract", "row_id": victim}]).version
        m2, recs2 = self.build(v2, {"g0": 5}, self.path("ds2"))
        self.assertEqual([r["row_id"] for r in recs2], expect)
        self.assertNotIn(victim, [r["row_id"] for r in recs2])
        # The retracted body is still readable at the old version.
        self.assertIsNotNone(self.svc.get_row(v1, victim))

    def test_regroup_moves_quota_between_groups(self):
        ops = [add(i, group=("A" if i < 20 else "B")) for i in range(40)]
        v1 = self.svc.apply_batch(ops).version
        build_dataset(self.svc, v1, self.seed, {"A": 4, "B": 4},
                      self.path("ds1"))
        # Move a top-A row into B.
        a_rows = self.svc.get_group_rows(v1, "A")
        cands = [make_candidate(r["row_id"], r["group"], r["weight"],
                                r["body_sha256"], self.seed) for r in a_rows]
        mover = select_top(cands, 1)[0].row_id
        v2 = self.svc.apply_batch(
            [{"op": "set_group", "row_id": mover, "group": "B"}]).version
        m2, recs2 = self.build(v2, {"A": 4, "B": 4}, self.path("ds2"))
        got = {}
        for r in recs2:
            got.setdefault(r["group"], []).append(r["row_id"])
        self.assertNotIn(mover, got["A"])
        self.assertIn(mover, got["B"])
        # Full rebuild equals a fresh exact computation of both groups.
        meta = self.svc.all_rows_meta(v2)
        ref = select_reference(self.seed, meta, {"A": 4, "B": 4})
        self.assertEqual(got["A"], ref["A"])
        self.assertEqual(got["B"], ref["B"])

    def test_refresh_equals_full_rebuild_complex(self):
        v1 = self.svc.apply_batch([add(i) for i in range(100)]).version
        quotas = {f"g{i}": 5 for i in range(5)}
        build_dataset(self.svc, v1, self.seed, quotas, self.path("old"))
        # A mixed batch: retract a selected row, regroup, reweight, rebody,
        # add new rows, and move a shard.
        a_rows = self.svc.get_group_rows(v1, "g0")
        cands = [make_candidate(r["row_id"], r["group"], r["weight"],
                                r["body_sha256"], self.seed) for r in a_rows]
        victim = select_top(cands, 1)[0].row_id
        v2 = self.svc.apply_batch([
            {"op": "retract", "row_id": victim},
            {"op": "set_group", "row_id": "r0001", "group": "g4"},
            {"op": "set_weight", "row_id": "r0002", "weight": 100},
            {"op": "set_body", "row_id": "r0003", "body": "corrected"},
            {"op": "set_shard", "row_id": "r0004", "shard": "zzz"},
            add(2000, group="g0"), add(2001, group="g3"),
        ]).version
        m_full, recs_full = build_dataset(
            self.svc, v2, self.seed, quotas, self.path("full"))
        m_ref, recs_ref, report = self.refresh(
            self.path("old"), new_version=v2, out=self.path("refresh"))
        self.assertEqual(compare_datasets(self.path("full"),
                                          self.path("refresh"))["equal"], True)
        self.assertEqual(report["new_version"], v2)
        self.assertIn("g0", report["rescanned_groups"])
        self.assertIn("g4", report["rescanned_groups"])
        self.assertIn("g2", report["rescanned_groups"])  # weight edit
        # And the whole thing verifies against the source + a fresh rebuild.
        self.assertTrue(verify_dataset(self.svc, self.path("refresh"))["ok"])

    def test_refresh_reuses_untouched_groups(self):
        v1 = self.svc.apply_batch([add(i) for i in range(60)]).version
        quotas = {f"g{i}": 4 for i in range(5)}
        build_dataset(self.svc, v1, self.seed, quotas, self.path("old"))
        v2 = self.svc.apply_batch(
            [{"op": "set_body", "row_id": "r0000", "body": "z"}]).version
        m, recs, report = self.refresh(self.path("old"), new_version=v2,
                                       out=self.path("new"))
        self.assertEqual(report["rescanned_groups"], ["g0"])
        self.assertEqual(report["reused_groups"], ["g1", "g2", "g3", "g4"])
        # Proven equal to full rebuild.
        build_dataset(self.svc, v2, self.seed, quotas, self.path("full"))
        self.assertTrue(compare_datasets(self.path("new"),
                                         self.path("full"))["equal"])

    def test_refresh_quota_change(self):
        v1 = self.svc.apply_batch([add(i) for i in range(40)]).version
        build_dataset(self.svc, v1, self.seed, {"g0": 2, "g1": 2},
                      self.path("old"))
        v2 = self.svc.apply_batch(
            [{"op": "set_body", "row_id": "r0000", "body": "x"}]).version
        m, recs, report = self.refresh(
            self.path("old"), new_version=v2,
            quotas={"g0": 5, "g1": 2}, out=self.path("new"))
        self.assertIn("g0", report["rescanned_groups"])
        self.assertEqual(len([r for r in recs if r["group"] == "g0"]), 5)
        build_dataset(self.svc, v2, self.seed, {"g0": 5, "g1": 2},
                      self.path("full"))
        self.assertTrue(compare_datasets(self.path("new"),
                                         self.path("full"))["equal"])

    def test_refresh_unchanged_version_is_identity(self):
        v1 = self.svc.apply_batch([add(i) for i in range(30)]).version
        build_dataset(self.svc, v1, self.seed, {f"g{i}": 3 for i in range(5)},
                      self.path("old"))
        m, recs, report = self.refresh(self.path("old"), new_version=v1,
                                       out=self.path("new"))
        self.assertEqual(report["rescanned_groups"], [])
        self.assertTrue(compare_datasets(self.path("old"),
                                         self.path("new"))["equal"])


class DatasetAuditTest(DatasetBase):
    def test_verify_detects_tampering(self):
        v = self.svc.apply_batch([add(i, group="g0") for i in range(20)]).version
        self.build(v, {"g0": 4}, self.path("ds"))
        # Tamper with a body in the JSONL.
        path = self.path("ds/samples.jsonl")
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        rec = json.loads(lines[0])
        rec["body"] = "TAMPERED"
        lines[0] = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        with self.assertRaises(RuntimeError):
            verify_dataset(self.svc, self.path("ds"))

    def test_verify_detects_wrong_order(self):
        v = self.svc.apply_batch([add(i, group="g0") for i in range(20)]).version
        self.build(v, {"g0": 4}, self.path("ds"))
        path = self.path("ds/samples.jsonl")
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        with self.assertRaises(RuntimeError):
            verify_dataset(self.svc, self.path("ds"))

    def test_verify_ok(self):
        v = self.svc.apply_batch([add(i) for i in range(30)]).version
        self.build(v, {f"g{i}": 3 for i in range(5)}, self.path("ds"))
        out = verify_dataset(self.svc, self.path("ds"))
        self.assertTrue(out["ok"])
        self.assertEqual(out["records"], 15)

    def test_every_selected_row_requeryable(self):
        v = self.svc.apply_batch([add(i) for i in range(50)]).version
        m, recs = self.build(v, {f"g{i}": 4 for i in range(5)}, self.path("ds"))
        fetched = self.svc.get_rows(v, [r["row_id"] for r in recs])
        self.assertEqual(len(fetched), len(recs))
        for r in recs:
            src = fetched[r["row_id"]]
            self.assertEqual(src["group"], r["group"])
            self.assertEqual(src["weight"], r["weight"])
            self.assertEqual(src["body"], r["body"])
            self.assertEqual(src["body_sha256"], r["body_sha256"])


class DatasetReopenTest(DatasetBase):
    def test_reopen_from_disk(self):
        path = os.path.join(self.dir, "src.sqlite")
        v1 = self.svc.apply_batch([add(i) for i in range(30)]).version
        build_dataset(self.svc, v1, self.seed, {"g0": 4}, self.path("ds"))
        # Close and reopen a fresh service object against the same file.
        svc2 = SourceService(path)
        self.assertEqual(svc2.latest_version(), v1)
        self.assertEqual(svc2.group_counts(v1), self.svc.group_counts(v1))
        m, recs = load_dataset(self.path("ds"))
        self.assertEqual(m["source_version"], v1)
        self.assertTrue(verify_dataset(svc2, self.path("ds"))["ok"])
        # Continue revisions with the reopened handle.
        v2 = svc2.apply_batch(
            [{"op": "retract", "row_id": "r0000"}]).version
        self.assertEqual(v2, v1 + 1)
        self.assertEqual(svc2.verify_bodies(v1), 30)


if __name__ == "__main__":
    unittest.main()
