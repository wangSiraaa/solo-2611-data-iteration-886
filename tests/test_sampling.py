import hashlib
import json
import os
import random
import shutil
import tempfile
import unittest
from decimal import Decimal, localcontext

from wrsource.sampling import (Candidate, boundary_certificate, compare_priority,
                               compare_rank, exact_tie, log_u, lottery_a,
                               lottery_hash, make_candidate, select_top,
                               sort_candidates)
from wrsource.reference import (brute_rank, exact_cross_compare,
                                select_reference)


def ref_hash(seed, row_id):
    rid = row_id.encode("utf-8")
    return hashlib.sha256(seed + len(rid).to_bytes(8, "big") + rid).digest()


class LotteryKeyTest(unittest.TestCase):
    def test_hash_construction(self):
        seed = bytes(range(32))
        for rid in ["a", "r-0001", "é", "行-42", "x" * 50]:
            expect = ref_hash(seed, rid)
            self.assertEqual(lottery_hash(seed, rid), expect)
            self.assertEqual(lottery_a(seed, rid),
                             int.from_bytes(expect, "big") + 1)

    def test_a_in_range(self):
        seed = b"\x01" * 32
        D = (1 << 256) + 1
        for i in range(500):
            a = lottery_a(seed, f"row-{i}")
            self.assertGreaterEqual(a, 1)
            self.assertLess(a, D + 1)
            self.assertLessEqual(a, D)

    def test_seed_changes_key(self):
        rids = [f"r{i}" for i in range(30)]
        k1 = [lottery_a(b"\x00" * 32, r) for r in rids]
        k2 = [lottery_a(b"\x01" * 32, r) for r in rids]
        self.assertNotEqual(k1, k2)

    def test_key_bound_only_to_row_id(self):
        # Same seed + row_id => same key regardless of any other field.
        seed = b"\x02" * 32
        c1 = make_candidate("row-x", "A", 1, "b1", seed)
        c2 = make_candidate("row-x", "B", 100, "b2", seed)
        self.assertEqual(c1.a, c2.a)

    def test_reference_matches_production_order(self):
        rng = random.Random(11)
        seed = bytes(rng.randrange(256) for _ in range(32))
        rows = [{"row_id": f"row-{i:03d}", "group": "g",
                 "weight": rng.randint(1, 100)} for i in range(60)]
        prod = [c.row_id for c in sort_candidates(
            [make_candidate(r["row_id"], r["group"], r["weight"], "", seed)
             for r in rows])]
        ref = [r["row_id"] for r in brute_rank(seed, rows)]
        self.assertEqual(prod, ref)

    def test_selection_matches_independent_reference(self):
        rng = random.Random(12)
        seed = bytes(rng.randrange(256) for _ in range(32))
        rows = []
        for g in ["A", "B", "C"]:
            for i in range(20):
                rows.append({"row_id": f"{g}-r{i:02d}", "group": g,
                             "weight": rng.randint(1, 100)})
        quotas = {"A": 7, "B": 0, "C": 25}
        cands = [make_candidate(r["row_id"], r["group"], r["weight"], "",
                                seed) for r in rows]
        got = {}
        for g in ["A", "B", "C"]:
            got[g] = [c.row_id for c in
                      select_top([c for c in cands if c.group == g],
                                 quotas[g])]
        ref = select_reference(seed, rows, quotas)
        self.assertEqual(got, ref)

    def test_weight_changes_priority(self):
        seed = b"\x03" * 32
        a = lottery_a(seed, "fixed-row")
        lo = Candidate("fixed-row", "g", 1, "", a, log_u(a, 1))
        hi = Candidate("fixed-row", "g", 100, "", a, log_u(a, 100))
        self.assertLess(compare_priority(lo, hi), 0)

    def test_same_key_same_weight_ties_on_row_id(self):
        seed = b"\x04" * 32
        a = 123456789
        c1 = Candidate("b", "g", 5, "", a, log_u(a, 5))
        c2 = Candidate("a", "g", 5, "", a, log_u(a, 5))
        self.assertTrue(exact_tie(c1, c2))
        self.assertLess(compare_rank(c2, c1), 0)  # "a" before "b"
        self.assertEqual([c.row_id for c in sort_candidates([c1, c2])],
                         ["a", "b"])

    def test_random_order_is_antireflexive_and_total(self):
        rng = random.Random(13)
        seed = bytes(rng.randrange(256) for _ in range(32))
        cands = [make_candidate(f"r{i}", "g", rng.randint(1, 100), "", seed)
                 for i in range(200)]
        for _ in range(500):
            c1, c2 = rng.sample(cands, 2)
            forward = compare_rank(c1, c2)
            backward = compare_rank(c2, c1)
            self.assertEqual(forward, -backward)
            self.assertNotEqual(forward, 0)


class ExactNearTieTest(unittest.TestCase):
    """Priorities that tie at float precision must still compare exactly."""

    @staticmethod
    def _craft_near_tie(a1, w1, w2, rng):
        # Choose a2 so that (a2/D)**(1/w2) is as close as possible to
        # (a1/D)**(1/w1), i.e. a2 ~= D * (a1/D)**(w2/w1).
        D = Decimal(2) ** 256 + 1
        with localcontext() as ctx:
            ctx.prec = 90
            target = (Decimal(a1) / D) ** (Decimal(w2) / Decimal(w1))
            a2 = int(target * D)
        for delta in range(-3, 4):
            cand = a2 + delta
            if 1 <= cand <= int(D):
                yield cand

    def test_float_screen_cannot_hide_exact_difference(self):
        rng = random.Random(21)
        checked = 0
        for _ in range(400):
            a1 = rng.getrandbits(256) | 1
            w1 = rng.randint(1, 100)
            w2 = rng.randint(1, 100)
            if w1 == w2:
                continue
            for a2 in self._craft_near_tie(a1, w1, w2, rng):
                c1 = Candidate("L", "g", w1, "", a1, log_u(a1, w1))
                c2 = Candidate("R", "g", w2, "", a2, log_u(a2, w2))
                got = compare_priority(c1, c2)
                want = exact_cross_compare(a1, w1, a2, w2)
                self.assertEqual(got, want,
                                 msg=f"a1={a1} w1={w1} a2={a2} w2={w2}")
                if abs(c1.lkey - c2.lkey) < 1e-9:
                    checked += 1
                # When the float screen is blind, the exact path must agree.
                if c1.lkey == c2.lkey:
                    self.assertNotEqual(got, 0)
        self.assertGreater(checked, 0, "no near-float tie was exercised")

    def test_tiny_constructed_gap(self):
        # One unit change in a is far below float resolution near 2**256.
        a1 = (1 << 255) + 12345
        a2 = a1 + 1
        c1 = Candidate("L", "g", 7, "", a1, log_u(a1, 7))
        c2 = Candidate("R", "g", 7, "", a2, log_u(a2, 7))
        self.assertEqual(c1.lkey, c2.lkey)      # float cannot see the gap
        self.assertLess(compare_priority(c1, c2), 0)  # exact sees a1 < a2
        self.assertFalse(exact_tie(c1, c2))

    def test_exact_cross_product_matches_huge_precision_decimal(self):
        rng = random.Random(22)
        D = Decimal(2) ** 256 + 1
        with localcontext() as ctx:
            ctx.prec = 220
            for _ in range(120):
                a1 = rng.getrandbits(256) | 1
                a2 = rng.getrandbits(256) | 1
                w1 = rng.randint(1, 100)
                w2 = rng.randint(1, 100)
                l_d = D.ln()
                p1 = ((Decimal(a1).ln() - l_d) / w1)
                p2 = ((Decimal(a2).ln() - l_d) / w2)
                d = p1 - p2
                if abs(d) < Decimal("1e-160"):
                    continue
                sign = (d > 0) - (d < 0)
                c1 = Candidate("L", "g", w1, "", a1, log_u(a1, w1))
                c2 = Candidate("R", "g", w2, "", a2, log_u(a2, w2))
                self.assertEqual(compare_priority(c1, c2), sign)

    def test_boundary_certificate(self):
        seed = b"\x05" * 32
        cands = [make_candidate(f"r{i}", "g", (i % 100) + 1, "", seed)
                 for i in range(40)]
        ordered = sort_candidates(cands)
        cert = boundary_certificate(ordered[4], ordered[5])
        self.assertGreater(int(cert["left_exact"]), int(cert["right_exact"]))


if __name__ == "__main__":
    unittest.main()
