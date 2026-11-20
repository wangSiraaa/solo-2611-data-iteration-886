import hashlib
import json
import os
import random
import shutil
import tempfile
import unittest

from wrsource.source import (BatchError, SourceService, VersionError,
                             VersionReader, decode_cursor, encode_cursor)


def make_row(i, group=None, weight=None, shard=None, body=None):
    return {
        "op": "add",
        "row_id": f"r{i:05d}",
        "group": group if group is not None else f"g{i % 7}",
        "weight": weight if weight is not None else (i % 100) + 1,
        "body": body if body is not None else f"body::{i}",
        "shard": shard if shard is not None else f"s{i % 3}",
    }


class SourceServiceTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.svc = SourceService(os.path.join(self.dir, "src.sqlite"))
        self.svc.init()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _seed(self, n=60):
        res = self.svc.apply_batch([make_row(i) for i in range(n)], note="seed")
        return res.version

    def test_initial_version_is_zero_and_empty(self):
        self.assertEqual(self.svc.latest_version(), 0)
        self.assertEqual(self.svc.group_counts(0), {})
        self.assertEqual(self.svc.read_all(0), [])

    def test_batch_atomic_and_final_state(self):
        # add then retract in one batch => never appears
        res = self.svc.apply_batch([
            make_row(1, group="A"),
            {"op": "retract", "row_id": "r00001"},
        ])
        self.assertIn("r00001", res.discarded_row_ids)
        self.assertEqual(self.svc.group_counts(res.version), {})
        # full rollback on bad op
        before = self.svc.latest_version()
        with self.assertRaises(BatchError):
            self.svc.apply_batch([make_row(2), {"op": "retract",
                                                "row_id": "missing"}])
        self.assertEqual(self.svc.latest_version(), before)
        self.assertIsNone(self.svc.current_row("r00002"))

    def test_revision_increments_on_logical_change_only(self):
        v = self._seed(5)
        r0 = self.svc.current_row("r00000")
        self.assertEqual(r0["rev"], 1)
        # pure shard move: no version bump, no rev bump
        res = self.svc.apply_batch(
            [{"op": "set_shard", "row_id": "r00000", "shard": "zzz"}])
        self.assertFalse(res.created)
        self.assertEqual(self.svc.latest_version(), v)
        self.assertEqual(self.svc.current_row("r00000")["rev"], 1)
        self.assertEqual(self.svc.current_row("r00000")["shard"], "zzz")
        # weight change: rev bump and new version
        res2 = self.svc.apply_batch(
            [{"op": "set_weight", "row_id": "r00000", "weight": 99}])
        self.assertTrue(res2.created)
        self.assertEqual(self.svc.current_row("r00000")["rev"], 2)
        self.assertEqual(res2.version, v + 1)

    def test_multi_op_same_row_collapses_to_final(self):
        v = self._seed(3)
        res = self.svc.apply_batch([
            {"op": "set_weight", "row_id": "r00000", "weight": 5},
            {"op": "set_weight", "row_id": "r00000", "weight": 50},
            {"op": "set_group", "row_id": "r00000", "group": "Z"},
        ])
        r = self.svc.current_row("r00000")
        self.assertEqual((r["weight"], r["group"]), (50, "Z"))
        self.assertEqual(r["rev"], 2)  # one logical revision for the batch
        self.assertEqual(res.logical_row_ids, ["r00000"])

    def test_multi_op_returns_to_original_is_nochange(self):
        self._seed(3)
        v = self.svc.latest_version()
        rev = self.svc.current_row("r00000")["rev"]
        res = self.svc.apply_batch([
            {"op": "set_weight", "row_id": "r00000", "weight": 77},
            {"op": "set_weight", "row_id": "r00000",
             "weight": self.svc.current_row("r00000")["weight"]},
        ])
        self.assertFalse(res.created)
        self.assertEqual(self.svc.latest_version(), v)
        self.assertEqual(self.svc.current_row("r00000")["rev"], rev)

    def test_add_retracted_row_reappears(self):
        v = self._seed(3)
        self.svc.apply_batch([{"op": "retract", "row_id": "r00001"}])
        self.assertIsNone(self.svc.get_row(self.svc.latest_version(),
                                           "r00001"))
        res = self.svc.apply_batch([{
            "op": "add", "row_id": "r00001", "group": "A", "weight": 3,
            "body": "back", "shard": "s0"}])
        got = self.svc.get_row(res.version, "r00001")
        self.assertEqual(got["body"], "back")
        self.assertGreater(self.svc.current_row("r00001")["rev"], 0)

    def test_duplicate_add_rejected(self):
        self._seed(2)
        with self.assertRaises(BatchError):
            self.svc.apply_batch([make_row(0)])

    def test_weight_bounds(self):
        for bad in [0, 101, -1, True, 3.0]:
            with self.assertRaises(BatchError):
                self.svc.apply_batch([make_row(1, weight=bad)])

    def test_stable_row_id_and_identity_across_shard_move(self):
        v1 = self._seed(10)
        before = {r["row_id"]: r for r in self.svc.read_all(v1)}
        # Move every row to new shards.
        res = self.svc.apply_batch(
            [{"op": "set_shard", "row_id": rid, "shard": "moved"}
             for rid in before])
        self.assertFalse(res.created)
        self.assertEqual(self.svc.latest_version(), v1)
        after = {r["row_id"]: r for r in self.svc.read_all(v1)}
        for rid, row in before.items():
            self.assertEqual(after[rid]["body"], row["body"])
            self.assertEqual(after[rid]["body_sha256"], row["body_sha256"])
            self.assertEqual(after[rid]["group"], row["group"])
            self.assertEqual(after[rid]["weight"], row["weight"])
        self.assertEqual(set(self.svc.shard_map(v1).values()), {"moved"})

    def test_old_versions_keep_readable_bodies(self):
        v1 = self._seed(10)
        v2 = self.svc.apply_batch(
            [{"op": "set_body", "row_id": "r00000", "body": "NEW-BODY"}]
        ).version
        self.assertEqual(self.svc.get_row(v1, "r00000")["body"], "body::0")
        self.assertEqual(self.svc.get_row(v2, "r00000")["body"], "NEW-BODY")
        self.assertEqual(self.svc.verify_bodies(v1), 10)
        self.assertEqual(self.svc.verify_bodies(v2), 10)

    def test_pagination_no_miss_no_dup_with_concurrent_updates(self):
        v1 = self._seed(103)
        seen = []
        reader = self.svc.open_reader(v1, page_size=7)
        cursor = None
        page_no = 0
        while True:
            page = reader.page(cursor) if page_no else reader.page()
            page_no += 1
            seen.extend(r["row_id"] for r in page["rows"])
            # Interleave a writer that changes rows in the *current* state.
            self.svc.apply_batch(
                [{"op": "set_weight", "row_id": f"r{i:05d}",
                  "weight": ((i + page_no) % 100) + 1}
                 for i in range(min(5, 103))])
            if page["done"]:
                break
            cursor = page["next_cursor"]
        self.assertEqual(len(seen), 103)
        self.assertEqual(len(set(seen)), 103)
        self.assertEqual(seen, sorted(seen))

    def test_pagination_uses_version_cursor(self):
        v1 = self._seed(20)
        page = self.svc.read_page(v1, limit=5)
        cur = page["next_cursor"]
        self.assertIsNotNone(cur)
        v, after = decode_cursor(cur)
        self.assertEqual(v, v1)
        v2 = self.svc.apply_batch(
            [{"op": "retract", "row_id": "r00000"}]).version
        # Using a v1 cursor against v2 must be refused by the reader.
        reader = self.svc.open_reader(v2, page_size=5)
        with self.assertRaises(VersionError):
            reader.page(cur)

    def test_page_order_matches_codepoints(self):
        v = self.svc.apply_batch([
            make_row(1), make_row(2),
            {"op": "add", "row_id": "Zeta", "group": "A", "weight": 1,
             "body": "z", "shard": "s"},
            {"op": "add", "row_id": "alpha", "group": "A", "weight": 1,
             "body": "a", "shard": "s"},
            {"op": "add", "row_id": "é", "group": "A", "weight": 1,
             "body": "e", "shard": "s"},
        ]).version
        ids = [r["row_id"] for r in self.svc.read_all(v)]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(ids[-1], "é")

    def test_diff_versions(self):
        v1 = self._seed(6)
        v2 = self.svc.apply_batch([
            {"op": "retract", "row_id": "r00000"},
            {"op": "set_group", "row_id": "r00001", "group": "moved"},
            {"op": "set_body", "row_id": "r00002", "body": "x"},
        ]).version
        d = self.svc.diff_versions(v1, v2)
        self.assertEqual(d["removed"], ["r00000"])
        self.assertEqual([c["row_id"] for c in d["changed"]],
                         ["r00001", "r00002"])

    def test_unknown_version_and_row(self):
        self._seed(3)
        with self.assertRaises(VersionError):
            self.svc.read_page(9999)
        with self.assertRaises(BatchError):
            self.svc.apply_batch([{"op": "set_weight", "row_id": "nope",
                                   "weight": 5}])


if __name__ == "__main__":
    unittest.main()


class SourceEdgeCaseTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.svc = SourceService(os.path.join(self.dir, "src.sqlite"))
        self.svc.init()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_add_after_retract_in_same_batch(self):
        v = self.svc.apply_batch([make_row(1, group="A", weight=5)]).version
        res = self.svc.apply_batch([
            {"op": "retract", "row_id": "r00001"},
            {"op": "add", "row_id": "r00001", "group": "B", "weight": 9,
             "body": "again", "shard": "sX"},
        ])
        self.assertTrue(res.created)
        got = self.svc.get_row(res.version, "r00001")
        self.assertEqual((got["group"], got["weight"], got["body"]),
                         ("B", 9, "again"))
        self.assertEqual(self.svc.current_row("r00001")["rev"], 2)

    def test_add_then_retract_then_add_new_row(self):
        res = self.svc.apply_batch([
            make_row(1, group="A"),
            {"op": "retract", "row_id": "r00001"},
            make_row(1, group="A"),
        ])
        self.assertEqual(res.discarded_row_ids, [])
        got = self.svc.get_row(res.version, "r00001")
        self.assertIsNotNone(got)

    def test_retract_twice_is_idempotent(self):
        v = self.svc.apply_batch([make_row(1)]).version
        self.svc.apply_batch([{"op": "retract", "row_id": "r00001"}])
        v2 = self.svc.latest_version()
        res = self.svc.apply_batch([{"op": "retract", "row_id": "r00001"}])
        self.assertFalse(res.created)
        self.assertEqual(self.svc.latest_version(), v2)
        self.assertEqual(self.svc.latest_version(), v + 1)

    def test_missing_fields_rejected(self):
        with self.assertRaises(BatchError):
            self.svc.apply_batch([{"op": "add", "row_id": "r1", "group": "A",
                                   "weight": 5, "shard": "s"}])  # no body
        with self.assertRaises(BatchError):
            self.svc.apply_batch([{"op": "add", "row_id": "r1", "group": "A",
                                   "weight": 5, "body": "b"}])  # no shard
        self.assertEqual(self.svc.group_counts(0), {})

    def test_body_with_newlines_and_unicode_roundtrip(self):
        body = "línea1\nlínea2\t中文 end"
        v = self.svc.apply_batch([
            make_row(1, group="A", body=body)]).version
        self.assertEqual(self.svc.get_row(v, "r00001")["body"], body)

    def test_empty_body_allowed(self):
        v = self.svc.apply_batch([make_row(1, group="A", body="")]).version
        self.assertEqual(self.svc.get_row(v, "r00001")["body"], "")

    def test_large_page_size_and_exact_boundary(self):
        v = self.svc.apply_batch([make_row(i) for i in range(10)]).version
        page = self.svc.read_page(v, limit=10)
        self.assertEqual(len(page["rows"]), 10)
        self.assertTrue(page["done"])
        self.assertIsNone(page["next_cursor"])
        page = self.svc.read_page(v, limit=9)
        self.assertFalse(page["done"])
        self.assertIsNotNone(page["next_cursor"])
