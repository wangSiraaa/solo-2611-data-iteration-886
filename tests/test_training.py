"""Tests for the append-only training exposure ledger."""

import os
import tempfile
import unittest

from wrsource.training import TrainingLedger, TrainingError


class TrainingLedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.led = TrainingLedger(
            os.path.join(self.tmp.name, "training.sqlite"))
        self.led.init()
        self.led.create_plan("T")

    def tearDown(self):
        self.tmp.cleanup()

    def test_batches_and_as_of_queries(self):
        s1 = self.led.register_batch("T", ["X", "Y"], source_version=1,
                                     evidence_revision=0, note="batch 1")
        s2 = self.led.register_batch("T", ["Z"], note="batch 2")
        self.assertEqual((s1, s2), (1, 2))
        self.assertEqual(self.led.exposed_rows("T"),
                         ["X", "Y", "Z"])
        self.assertEqual(self.led.exposed_rows("T", as_of_batch=1),
                         ["X", "Y"])
        self.assertTrue(self.led.is_exposed("T", "Z"))
        self.assertFalse(self.led.is_exposed("T", "Z", as_of_batch=1))

    def test_retracted_record_stays_an_exposure(self):
        self.led.register_batch("T", ["X"])
        self.led.retract_exposure("T", "X")
        # Default view keeps history: the original was seen.
        self.assertEqual(self.led.exposed_rows("T"), ["X"])
        self.assertEqual(
            self.led.exposed_rows("T", include_retracted=False), [])
        state = self.led.plan_state("T")
        self.assertEqual(state.exposed, [])
        self.assertEqual(state.retracted, ["X"])
        with self.assertRaises(TrainingError):
            self.led.retract_exposure("T", "never-shipped")

    def test_unknown_plan_and_empty_batch(self):
        with self.assertRaises(TrainingError):
            self.led.register_batch("nope", ["X"])
        with self.assertRaises(TrainingError):
            self.led.register_batch("T", [])

    def test_duplicate_emission_keeps_first_batch_attribution(self):
        self.led.register_batch("T", ["X"])
        self.led.register_batch("T", ["X", "Y"])
        rows = {r.row_id: r.batch_seq
                for b in self.led.batches("T") for r in [type("x", (), {
                    "row_id": rid, "batch_seq": b.batch_seq})()
                    for rid in b.row_ids]}
        self.assertEqual(set(rows), {"X", "Y"})
        # X remains a member of batch 1's unique exposure set.
        self.assertEqual(self.led.exposed_rows("T", as_of_batch=1), ["X"])

    def test_independent_plans(self):
        self.led.create_plan("U")
        self.led.register_batch("T", ["X"])
        self.led.register_batch("U", ["P"])
        self.assertEqual(self.led.exposed_rows("T"), ["X"])
        self.assertEqual(self.led.exposed_rows("U"), ["P"])
        self.assertEqual(sorted(self.led.list_plans()), ["T", "U"])


if __name__ == "__main__":
    unittest.main()
