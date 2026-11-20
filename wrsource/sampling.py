"""Exact lottery-key arithmetic and deterministic top-quota selection.

The sampling rule implemented here is the one fixed by the task:

    h = SHA-256(seed[32] || uint64_be(len(utf8(row_id))) || utf8(row_id))
    a = int(h) + 1                    # so u = a / D,  D = 2**256 + 1
    priority = u ** (1 / weight)      # weight is an integer in 1..100

Rows are ranked by priority descending; exact priority ties are broken by
row_id in Unicode code-point order.  "Exact tie" means the two real-valued
priorities are equal, not that a float rounded them to the same value.

Comparison derivation (all quantities positive):

    priority1 > priority2
  <=> (a1/D)**(1/w1) > (a2/D)**(1/w2)
  <=> (raise both sides to w1*w2, order preserving)
      (a1/D)**w2 > (a2/D)**w1
  <=> a1**w2 * D**w1 > a2**w1 * D**w2          (exact integer compare)

A float log is used only as a *screen*.  It may declare a winner only when
the gap safely exceeds a conservative error bound.  Boundary cases go through
60-digit Decimal logs and finally exact integer powers, so two extremely
close keys can never be collapsed by binary floating point.

Only the Python standard library is used.
"""

from __future__ import annotations

import hashlib
import math
import sys
from decimal import Decimal, localcontext
from functools import cmp_to_key
from typing import Dict, Iterable, List, NamedTuple, Sequence

# Exact cross products can exceed Python 3.11's default 4300-digit limit for
# int<->str conversion (a 100-weight, 256-bit key yields ~15k digits).  The
# conversion itself is only used for manifest evidence, never for ranking.
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(200_000)

BITS = 256
D = (1 << BITS) + 1
D_DEC = Decimal(D)
HALF_D = D >> 1

# math.log has absolute error far below 1e-12 over [1, D]; 1e-9 is safe.
FAST_LOG_EPS = 1e-9
# 60-digit Decimal logs have error ~1e-55; require a 1e-50 gap.
DECIMAL_EPS = Decimal("1e-50")

_U64 = 1 << 64
_LOG_D = math.log(D)


class Candidate(NamedTuple):
    """A row plus its exact lottery material (no float priority is stored)."""

    row_id: str
    group: str
    weight: int
    body_sha256: str
    a: int          # int(SHA-256) + 1, the exact numerator of u
    lkey: float     # log(u)/weight, a screen only

    @property
    def key_hex(self) -> str:
        """The 64-hex SHA-256 digest that produced ``a``."""
        return format(self.a - 1, "064x")


def lottery_hash(seed: bytes, row_id: str) -> bytes:
    """SHA-256 over seed || u64be(len(utf8(row_id))) || utf8(row_id)."""
    if len(seed) != 32:
        raise ValueError("seed must be exactly 32 bytes (256 bits)")
    rid = row_id.encode("utf-8")
    if len(rid) >= _U64:
        raise ValueError("row_id too long")
    return hashlib.sha256(seed + len(rid).to_bytes(8, "big") + rid).digest()


def lottery_a(seed: bytes, row_id: str) -> int:
    return int.from_bytes(lottery_hash(seed, row_id), "big") + 1


def log_u(a: int, weight: int) -> float:
    return (math.log(a) - _LOG_D) / weight


def make_candidate(row_id: str, group: str, weight: int, body_sha256: str,
                   seed: bytes) -> Candidate:
    if not 1 <= weight <= 100:
        raise ValueError(f"weight out of range for {row_id!r}: {weight}")
    a = lottery_a(seed, row_id)
    return Candidate(row_id, group, weight, body_sha256, a, log_u(a, weight))


def _compare_exact(a1: int, w1: int, a2: int, w2: int) -> int:
    """Exact sign of priority1 - priority2 via integer cross products."""
    s1 = pow(a1, w2) * pow(D, w1)
    s2 = pow(a2, w1) * pow(D, w2)
    return (s1 > s2) - (s1 < s2)


def _compare_decimal(a1: int, w1: int, a2: int, w2: int) -> int:
    """60-digit Decimal screen; 0 means 'too close, use exact integers'."""
    with localcontext() as ctx:
        ctx.prec = 60
        l_d = D_DEC.ln()
        d = (Decimal(a1).ln() - l_d) / w1 - (Decimal(a2).ln() - l_d) / w2
        if abs(d) > DECIMAL_EPS:
            return 1 if d > 0 else -1
    return 0


def compare_priority(c1: Candidate, c2: Candidate) -> int:
    """Exact 3-way comparison of priorities (-1/0/+1).

    0 is returned only when the real-valued priorities are exactly equal,
    in which case callers break the tie by row_id.
    """
    a1, w1 = c1.a, c1.weight
    a2, w2 = c2.a, c2.weight
    if w1 == w2:
        return (a1 > a2) - (a1 < a2)
    d = c1.lkey - c2.lkey
    if abs(d) > FAST_LOG_EPS:
        return 1 if d > 0 else -1
    r = _compare_decimal(a1, w1, a2, w2)
    if r:
        return r
    return _compare_exact(a1, w1, a2, w2)


def compare_rank(c1: Candidate, c2: Candidate) -> int:
    """Total selection order: priority descending, then row_id ascending."""
    r = compare_priority(c1, c2)
    if r:
        return -r
    if c1.row_id == c2.row_id:
        return 0
    return -1 if c1.row_id < c2.row_id else 1


def sort_candidates(cands: Iterable[Candidate]) -> List[Candidate]:
    """Return candidates in exact selection order (best first)."""
    return sorted(cands, key=cmp_to_key(compare_rank))


def select_top(cands: Iterable[Candidate], quota: int) -> List[Candidate]:
    """Select up to ``quota`` rows with the exact deterministic rule."""
    if quota <= 0:
        return []
    return sort_candidates(cands)[:quota]


def exact_tie(c1: Candidate, c2: Candidate) -> bool:
    """True iff the two real-valued priorities are exactly equal."""
    return compare_priority(c1, c2) == 0


def priority_material(c: Candidate) -> Dict[str, object]:
    """Per-row exact material: enough to recompute the exact comparison."""
    return {
        "hash_hex": c.key_hex,
        "a": str(c.a),
        "weight": c.weight,
        "log_screen": c.lkey,
        "priority_expr": f"(({c.a}/{D})^(1/{c.weight}))",
    }


def boundary_certificate(selected: Candidate, cut: Candidate) -> Dict[str, object]:
    """Exact proof that ``selected`` outranks the first unselected row.

    The two huge integers are the exact cross products from the derivation;
    the left one must be strictly greater.
    """
    left = pow(selected.a, cut.weight) * pow(D, selected.weight)
    right = pow(cut.a, selected.weight) * pow(D, cut.weight)
    if not left > right:
        raise AssertionError("boundary certificate is not strictly ordered")
    return {
        "selected": {
            "row_id": selected.row_id,
            "hash_hex": selected.key_hex,
            "a": str(selected.a),
            "weight": selected.weight,
        },
        "cut": {
            "row_id": cut.row_id,
            "hash_hex": cut.key_hex,
            "a": str(cut.a),
            "weight": cut.weight,
        },
        "rule": "a_sel^w_cut * D^w_sel > a_cut^w_sel * D^w_cut",
        "left_exact": str(left),
        "right_exact": str(right),
    }


def comparison_material(c1: Candidate, c2: Candidate) -> Dict[str, object]:
    """Full exact comparison record between two candidates (for tests/audit)."""
    sign = compare_priority(c1, c2)
    return {
        "left": {"row_id": c1.row_id, "hash_hex": c1.key_hex,
                 "a": str(c1.a), "weight": c1.weight},
        "right": {"row_id": c2.row_id, "hash_hex": c2.key_hex,
                  "a": str(c2.a), "weight": c2.weight},
        "left_cross_exact": str(pow(c1.a, c2.weight) * pow(D, c1.weight)),
        "right_cross_exact": str(pow(c2.a, c1.weight) * pow(D, c2.weight)),
        "priority_cmp": sign,
        "rank_cmp": (-sign if sign else
                     (-1 if c1.row_id < c2.row_id else
                      (1 if c1.row_id > c2.row_id else 0))),
    }
