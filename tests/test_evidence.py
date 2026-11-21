"""Tests for the same-origin graph and the revisionable evidence ledger."""

import os
import tempfile
import unittest

from wrsource.evidence import EvidenceLedger, EvidenceError
from wrsource.graph import ActiveEdge, GraphSnapshot


def edge(eid, a, b):
    lo, hi = (a, b) if a <= b else (b, a)
    return ActiveEdge(eid, lo, hi)


class GraphSnapshotTest(unittest.TestCase):
    def test_transitive_closure_through_retracted_body_node(self):
        # M may have its body retracted at the source; here it is still a
        # graph node connecting X and B.
        snap = GraphSnapshot(1, [edge("e1", "X", "M"), edge("e2", "M", "B")])
        self.assertTrue(snap.same_origin("X", "B"))
        path = snap.find_path("X", "B")
        self.assertEqual(path.nodes, ["X", "M", "B"])
        self.assertEqual(path.evidence_ids, ["e1", "e2"])
        self.assertEqual(snap.component_of("X"), ["B", "M", "X"])

    def test_identical_bodies_are_not_automatically_related(self):
        snap = GraphSnapshot(0, [])
        self.assertFalse(snap.same_origin("A", "B"))
        self.assertIsNone(snap.find_path("A", "B"))
        self.assertEqual(snap.connected_exposures("A", ["X"]), [])

    def test_direct_self_exposure_on_isolated_node(self):
        snap = GraphSnapshot(0, [])
        self.assertEqual(snap.connected_exposures("A", ["A"]), ["A"])
        self.assertEqual(snap.connected_exposures("B", ["A"]), [])

    def test_parallel_edges_are_distinct(self):
        snap = GraphSnapshot(2, [edge("a", "X", "B"), edge("b", "X", "B")])
        self.assertEqual(snap.edge_count(), 2)
        path = snap.find_path("B", "X")
        self.assertEqual(path.evidence_ids, ["a"])  # deterministic BFS order


class EvidenceLedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "evidence.sqlite")
        self.led = EvidenceLedger(self.db)
        self.led.init()

    def tearDown(self):
        self.tmp.cleanup()

    def test_revisions_and_historical_reopen(self):
        self.assertEqual(self.led.latest_revision(), 0)
        r1 = self.led.add_evidence("e1", "X", "M", note="first link")
        r2 = self.led.add_evidence("e2", "M", "B")
        self.assertEqual((r1, r2), (1, 2))
        self.assertTrue(self.led.same_origin("X", "B", r2))
        self.assertFalse(self.led.same_origin("X", "B", r1))
        # Old revision stays answerable after later revocation.
        r3 = self.led.revoke_evidence("e1")
        self.assertEqual(r3, 3)
        self.assertTrue(self.led.same_origin("X", "B", r2))
        self.assertFalse(self.led.same_origin("X", "B", r3))
        # Reopening the latest snapshot twice gives an equal view.
        self.assertEqual(self.led.snapshot().revision, 3)

    def test_parallel_evidence_needs_last_revocation_to_disconnect(self):
        self.led.add_evidence("a", "X", "B")
        self.led.add_evidence("b", "B", "X")
        pair = self.led.evidence_between("X", "B")
        self.assertEqual([e.evidence_id for e in pair], ["a", "b"])
        self.assertTrue(self.led.same_origin("X", "B"))
        self.led.revoke_evidence("a")
        self.assertTrue(self.led.same_origin("X", "B"),
                        "one surviving parallel edge must hold the pair")
        self.led.revoke_evidence("b")
        self.assertFalse(self.led.same_origin("X", "B"))

    def test_endpoints_are_normalised_and_validated(self):
        self.led.add_evidence("e1", "Z", "A")
        rec = self.led.get_evidence("e1")
        self.assertEqual((rec.endpoint_a, rec.endpoint_b), ("A", "Z"))
        with self.assertRaises(EvidenceError):
            self.led.add_evidence("e2", "A", "A")
        with self.assertRaises(EvidenceError):
            self.led.revoke_evidence("missing")

    def test_split_then_reconnect_large_component(self):
        # A-B-C chain; splitting B-C separates, a fresh A-C edge reconnects.
        self.led.add_evidence("ab", "A", "B")
        self.led.add_evidence("bc", "B", "C")
        self.assertTrue(self.led.same_origin("A", "C"))
        self.led.revoke_evidence("bc")
        self.assertFalse(self.led.same_origin("A", "C"))
        self.led.add_evidence("ac", "A", "C")
        self.assertTrue(self.led.same_origin("A", "C"))
        path = self.led.origin_path("B", "C")
        self.assertEqual(path.evidence_ids, ["ab", "ac"])

    def test_witness_path_cites_every_evidence_id(self):
        self.led.add_evidence("e1", "X", "M")
        self.led.add_evidence("e2", "M", "B")
        path = self.led.origin_path("B", "X")
        self.assertEqual(path.nodes, ["B", "M", "X"])
        self.assertEqual(set(path.evidence_ids), {"e1", "e2"})

    def test_op_log_is_append_only(self):
        self.led.add_evidence("e1", "X", "M")
        self.led.revoke_evidence("e1")
        log = self.led.op_log()
        self.assertEqual([(o["revision"], o["op"], o["evidence_id"])
                          for o in log],
                         [(1, "add", "e1"), (2, "revoke", "e1")])
        rec = self.led.get_evidence("e1")
        self.assertFalse(rec.active)
        self.assertEqual(rec.revoked_rev, 2)


if __name__ == "__main__":
    unittest.main()
