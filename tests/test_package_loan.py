"""End-to-end tests: eval packages, recheck/reissue and loan timelines.

The synthetic source mirrors the worked example in the task: group g1 holds
rows A and B with equal weight and an all-zero seed, and the lottery ranks B
ahead of A; row X lives in another group; M is a node whose body gets
retracted but still connects evidence edges.
"""

import os
import tempfile
import unittest

from functools import cmp_to_key

from wrsource.evidence import EvidenceLedger
from wrsource.loan import (LoanedPackage, investigate_loans, loan_timeline,
                           loaned_package_from_manifest)
from wrsource.package import (PackageError, PackageRegistry,
                              build_eval_package, explain_eligibility,
                              recheck_package, reissue_clean_package,
                              selected_row_ids)
from wrsource.sampling import make_candidate, compare_rank
from wrsource.source import SourceService
from wrsource.training import TrainingLedger

SEED = bytes(32)


def _add(svc, row_id, group, weight, body, shard="s0"):
    svc.apply_batch([{"op": "add", "row_id": row_id, "group": group,
                      "weight": weight, "body": body, "shard": shard}])


class PackageScenarioTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.svc = SourceService(os.path.join(d, "source.sqlite"))
        self.svc.init()
        _add(self.svc, "A", "g1", 50, "body alpha")
        _add(self.svc, "B", "g1", 50, "body beta")
        _add(self.svc, "M", "g1", 50, "body mmm")
        _add(self.svc, "X", "g2", 50, "body xray")
        _add(self.svc, "C", "g1", 50, "body gamma")
        self.version = self.svc.latest_version()

        self.evidence = EvidenceLedger(os.path.join(d, "evidence.sqlite"))
        self.evidence.init()

        self.training = TrainingLedger(os.path.join(d, "training.sqlite"))
        self.training.init()
        self.training.create_plan("T")
        self.training.create_plan("T-empty")
        # X was actually emitted under plan T.
        self.training.register_batch("T", ["X"], source_version=self.version,
                                     evidence_revision=0, note="expose X")
        self.registry = PackageRegistry(os.path.join(d, "packages.sqlite"))
        self.registry.init()
        self.quotas = {"g1": 1}

    def tearDown(self):
        self.tmp.cleanup()

    def test_lottery_ranks_b_first_with_zero_seed(self):
        cands = [make_candidate(r["row_id"], "g1", r["weight"],
                                r["body_sha256"], SEED)
                 for r in self.svc.group_rows_meta(self.version, "g1")]
        ordered = sorted(cands, key=cmp_to_key(compare_rank))
        a = next(c for c in ordered if c.row_id == "A")
        b = next(c for c in ordered if c.row_id == "B")
        self.assertLess(compare_rank(b, a), 0)

    # -- the three worked selections from the task ------------------------
    def _issue(self, plan, ev_rev, cutoff=None):
        return self.registry.issue(
            self.svc, self.version, SEED, self.quotas, self.training, plan,
            self.evidence, ev_rev, cutoff_batch=cutoff)

    def test_exposed_chain_through_retracted_m_forces_selection_of_a(self):
        # X--M--B while M's body is retracted; B must be excluded, A selected.
        self.evidence.add_evidence("xm", "X", "M")
        self.evidence.add_evidence("mb", "M", "B")
        self.svc.apply_batch([{"op": "retract", "row_id": "M"}])
        v2 = self.svc.latest_version()
        self.assertIsNone(self.svc.get_row(v2, "M"))
        # Evidence node M still connects X and B.
        self.assertTrue(self.evidence.same_origin("X", "B"))

        manifest = self.registry.issue(
            self.svc, v2, SEED, self.quotas, self.training, "T",
            self.evidence, self.evidence.latest_revision())
        self.assertEqual(selected_row_ids(manifest), ["A"])
        g1 = manifest["groups"]["g1"]
        # M is no longer a candidate (body retracted at the source), but its
        # evidence node still links B to the exposed X.
        self.assertEqual([e["row_id"] for e in g1["excluded"]], ["B"])
        b_reasons = next(e for e in g1["excluded"]
                         if e["row_id"] == "B")["reasons"]
        self.assertEqual(b_reasons[0]["path"]["nodes"],
                         ["X", "M", "B"])
        # Pins are frozen into the signed manifest.
        self.assertEqual(manifest["training_plan"], "T")
        self.assertEqual(manifest["exposure_rows_at_issue"], ["X"])

    def test_empty_plan_selects_b_with_unchanged_evidence(self):
        self.evidence.add_evidence("xm", "X", "M")
        self.evidence.add_evidence("mb", "M", "B")
        manifest = self._issue("T-empty", self.evidence.latest_revision())
        self.assertEqual(selected_row_ids(manifest), ["B"])
        self.assertEqual(manifest["groups"]["g1"]["excluded"], [])

    def test_revoking_unique_mb_edge_reselects_b_under_same_plan(self):
        self.evidence.add_evidence("xm", "X", "M")
        self.evidence.add_evidence("mb", "M", "B")
        self._issue("T", self.evidence.latest_revision())
        self.evidence.revoke_evidence("mb")
        manifest = self._issue("T", self.evidence.latest_revision())
        self.assertEqual(selected_row_ids(manifest), ["B"])
        # A and B are two distinct records even though their bodies could be
        # identical; body equality never creates a same-origin edge.
        self.assertFalse(self.evidence.same_origin("A", "B"))

    def test_same_body_does_not_imply_same_origin(self):
        # Give A and B literally identical bodies: still two row identities.
        self.svc.apply_batch([{"op": "set_body", "row_id": "A",
                               "body": "identical text"}])
        self.svc.apply_batch([{"op": "set_body", "row_id": "B",
                               "body": "identical text"}])
        v = self.svc.latest_version()
        rows = {r["row_id"]: r for r in self.svc.get_group_rows(v, "g1")}
        self.assertEqual(rows["A"]["body"], rows["B"]["body"])
        self.assertNotEqual(rows["A"]["row_id"], rows["B"]["row_id"])
        manifest = self.registry.issue(
            self.svc, v, SEED, self.quotas, self.training, "T",
            self.evidence, 0)
        self.assertEqual(selected_row_ids(manifest), ["B"])

    # -- eligibility explanation ------------------------------------------
    def test_explain_eligibility_is_reproducible_from_pins(self):
        self.evidence.add_evidence("xm", "X", "M")
        self.evidence.add_evidence("mb", "M", "B")
        snap = self.evidence.snapshot()
        exposed = {"X": False}
        no = explain_eligibility("B", exposed, snap)
        self.assertFalse(no["eligible"])
        self.assertEqual(no["reasons"][0]["path"]["evidence_ids"],
                         ["xm", "mb"])
        yes = explain_eligibility("A", exposed, snap)
        self.assertTrue(yes["eligible"])

    # -- signed package recheck + clean reissue ---------------------------
    def test_signed_package_recheck_lists_actual_rows_and_reissue_backfills(
            self):
        # Sign while X and B are separate: B wins the slot.
        signed = self._issue("T", 0)
        signed_id = signed["package_id"]
        self.assertEqual(selected_row_ids(signed), ["B"])

        # Later evidence links B to the exposed X; old package is untouched.
        self.evidence.add_evidence("xb", "X", "B")
        report = recheck_package(self.registry.open(signed_id),
                                 self.training, self.evidence)
        self.assertEqual([q["row_id"] for q in
                          report["quarantine_required"]], ["B"])
        self.assertEqual(report["still_clean"], [])
        witness = report["quarantine_required"][0]["reasons"][0]
        self.assertEqual(witness["exposed_row_id"], "X")
        self.assertEqual(witness["path"]["evidence_ids"], ["xb"])
        # The stored signed manifest still carries the old judgment.
        self.assertEqual(self.registry.open(signed_id)["evidence_revision"], 0)

        new_manifest, reissue = reissue_clean_package(
            self.registry, self.svc, signed_id, self.training,
            self.evidence)
        self.assertEqual(selected_row_ids(new_manifest), ["A"])
        self.assertEqual(new_manifest["parent_package_id"], signed_id)
        self.assertEqual([d["row_id"] for d in
                          reissue["dropped_rows"]], ["B"])
        self.assertEqual(reissue["backfilled_rows"], ["A"])
        self.assertTrue(reissue["old_package_immutable"])
        # Two distinct signed packages now coexist.
        self.assertEqual(len(self.registry.list_packages()), 2)

    def test_tampered_manifest_is_refused(self):
        manifest = build_eval_package(
            self.svc, self.version, SEED, self.quotas, self.training,
            "T-empty", self.evidence, 0)
        manifest["groups"]["g1"]["quota"] = 999
        with self.assertRaises(PackageError):
            self.registry.sign(manifest)

    def test_retracted_training_record_still_quarantines(self):
        self.evidence.add_evidence("xb", "X", "B")
        self.training.retract_exposure("T", "X")
        manifest = self._issue("T", self.evidence.latest_revision())
        self.assertEqual(selected_row_ids(manifest), ["A"])
        reason = manifest["groups"]["g1"]["excluded"][0]["reasons"][0]
        self.assertTrue(reason["training_record_retracted"])

    def test_parallel_edge_keeps_package_clean_until_last_revocation(self):
        r_a = self.evidence.add_evidence("a", "X", "B")
        r_b = self.evidence.add_evidence("b", "X", "B")
        self.assertEqual(selected_row_ids(self._issue("T", r_b)), ["A"])
        self.evidence.revoke_evidence("a")
        self.assertEqual(
            selected_row_ids(self._issue("T",
                                         self.evidence.latest_revision())),
            ["A"])
        self.evidence.revoke_evidence("b")
        self.assertEqual(
            selected_row_ids(self._issue("T",
                                         self.evidence.latest_revision())),
            ["B"])

    def test_old_evidence_revision_reopens_old_judgment(self):
        self.evidence.add_evidence("xb", "X", "B")
        new_rev = self.evidence.latest_revision()
        clean = self._issue("T", 0)
        dirty = self._issue("T", new_rev)
        self.assertEqual(selected_row_ids(clean), ["B"])
        self.assertEqual(selected_row_ids(dirty), ["A"])
        self.assertEqual(clean["evidence_revision"], 0)
        self.assertEqual(dirty["evidence_revision"], new_rev)

    def test_split_then_reconnect_group_is_recheckable_per_plan(self):
        # Many plans exist; a recheck for one plan must not see another's
        # exposures.
        self.training.create_plan("other")
        self.training.register_batch("other", ["Q"])
        signed = self._issue("T", 0)
        # Big component X-M-B linked, split, then reconnected via new edge.
        self.evidence.add_evidence("xm", "X", "M")
        self.evidence.add_evidence("mb", "M", "B")
        self.evidence.revoke_evidence("mb")
        report_mid = recheck_package(signed, self.training, self.evidence)
        self.assertEqual(report_mid["quarantine_required"], [])
        self.evidence.add_evidence("xb2", "X", "B")
        report_end = recheck_package(signed, self.training, self.evidence)
        self.assertEqual([q["row_id"] for q in
                          report_end["quarantine_required"]], ["B"])
        # Plan "other" never exposed anything connected to B.
        other_manifest = self.registry.issue(
            self.svc, self.version, SEED, self.quotas, self.training,
            "other", self.evidence, self.evidence.latest_revision())
        self.assertEqual(selected_row_ids(other_manifest), ["B"])


class LoanTimelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.svc = SourceService(os.path.join(d, "source.sqlite"))
        self.svc.init()
        _add(self.svc, "A", "g1", 50, "alpha")
        _add(self.svc, "B", "g1", 50, "beta")
        _add(self.svc, "X", "g2", 50, "xray")
        self.version = self.svc.latest_version()
        self.evidence = EvidenceLedger(os.path.join(d, "evidence.sqlite"))
        self.evidence.init()
        self.training = TrainingLedger(os.path.join(d, "training.sqlite"))
        self.training.init()
        self.training.create_plan("T")
        self.training.create_plan("no-x")
        self.training.register_batch("T", ["X"])
        self.registry = PackageRegistry(os.path.join(d, "packages.sqlite"))
        self.registry.init()
        self.pkg = self.registry.issue(
            self.svc, self.version, SEED, {"g1": 2}, self.training,
            "no-x", self.evidence, 0)
        self.loan = loaned_package_from_manifest("loan-1", self.pkg)

    def tearDown(self):
        self.tmp.cleanup()

    def test_enter_on_first_edge_exit_on_last_revocation_only(self):
        # Revision sequence from the task:
        # 1 add a (X-B), 2 add b (X-B parallel), 3 revoke a, 4 revoke b.
        self.evidence.add_evidence("a", "X", "B")
        self.evidence.add_evidence("b", "X", "B")
        self.evidence.revoke_evidence("a")
        self.evidence.revoke_evidence("b")
        result = loan_timeline(
            self.loan, "T", start_rev=0, end_rev=4,
            training=self.training, evidence=self.evidence, cutoff_batch=1)
        self.assertEqual(result.initial_quarantined, [])
        actions = [(e.revision, e.action, e.row_id) for e in result.events]
        self.assertEqual(actions, [(1, "enter", "B"), (4, "exit", "B")])
        self.assertEqual(result.final_quarantined, [])
        enter = result.events[0].detail["reasons"][0]
        self.assertEqual(enter["path"]["evidence_ids"], ["a"])
        self.assertEqual(result.events[1].detail["released_by_op"]["op"],
                         "revoke")
        # A shares no evidence path with X at any point.
        self.assertNotIn("A", result.initial_quarantined)

    def test_plan_without_exposure_has_no_actions(self):
        self.evidence.add_evidence("a", "X", "B")
        self.evidence.add_evidence("b", "X", "B")
        self.evidence.revoke_evidence("a")
        self.evidence.revoke_evidence("b")
        result = loan_timeline(
            self.loan, "no-x", 0, 4, self.training, self.evidence,
            cutoff_batch=1)
        self.assertEqual(result.events, [])
        self.assertEqual(result.initial_quarantined, [])

    def test_quarantine_already_active_at_window_start(self):
        self.evidence.add_evidence("a", "X", "B")          # rev 1
        self.evidence.add_evidence("b", "M", "X")          # rev 2
        result = loan_timeline(
            self.loan, "T", 2, 2, self.training, self.evidence,
            cutoff_batch=1)
        self.assertEqual(result.initial_quarantined, ["B"])
        self.assertEqual(result.events, [])
        # Reopening the earlier window start shows the pre-link state.
        early = loan_timeline(
            self.loan, "T", 0, 0, self.training, self.evidence,
            cutoff_batch=1)
        self.assertEqual(early.initial_quarantined, [])

    def test_bulk_investigation_shares_revision_sweep(self):
        self.evidence.add_evidence("a", "X", "B")
        second = LoanedPackage("loan-2", self.pkg["package_id"],
                               ["A", "B"])
        results = investigate_loans(
            [self.loan, second], "T", 0, 1, self.training,
            self.evidence, cutoff_batch=1)
        self.assertEqual(set(results), {"loan-1", "loan-2"})
        for res in results.values():
            self.assertEqual([(e.revision, e.action) for e in res.events],
                             [(1, "enter")])

    def test_exposure_after_cutoff_does_not_move_loan(self):
        # A second training batch inside the window must be ignored.
        self.training.register_batch("T", ["A"])
        self.assertEqual(
            loan_timeline(self.loan, "T", 0, 0, self.training,
                          self.evidence, cutoff_batch=1).final_quarantined,
            [])

    def test_chain_witness_through_intermediate_node_in_timeline(self):
        # X--M--B with M itself never exposed and not in the loan: the enter
        # event must still cite the full two-edge chain as its reason.
        self.evidence.add_evidence("xm", "X", "M")
        self.evidence.add_evidence("mb", "M", "B")
        result = loan_timeline(self.loan, "T", 0, 2, self.training,
                               self.evidence, cutoff_batch=1)
        enter = next(e for e in result.events if e.row_id == "B")
        reason = enter.detail["reasons"][0]
        self.assertEqual(reason["exposed_row_id"], "X")
        self.assertEqual(reason["path"]["nodes"], ["X", "M", "B"])
        self.assertEqual(reason["path"]["evidence_ids"], ["xm", "mb"])

    def test_multiple_exposed_rows_all_cited_as_reasons(self):
        # Two exposed rows in B's component: a window starting after both
        # edges exist cites both witnesses in the initial-state detail.
        self.training.register_batch("T", ["Y"])
        self.evidence.add_evidence("xb", "X", "B")
        self.evidence.add_evidence("yb", "Y", "B")
        result = loan_timeline(self.loan, "T", 2, 2, self.training,
                               self.evidence, cutoff_batch=2)
        self.assertEqual(result.initial_quarantined, ["B"])
        cited = sorted(r["exposed_row_id"]
                       for r in result.initial_detail["B"]["reasons"])
        self.assertEqual(cited, ["X", "Y"])
        # No transition occurs inside a zero-length window that starts dirty.
        self.assertEqual(result.events, [])


if __name__ == "__main__":
    unittest.main()
