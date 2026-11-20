"""Deterministic fallback reference sampler for small test sets.

This module is deliberately independent from :mod:`wrsource.sampling` in its
*implementation*: it uses exact fractional powers raised to 256*100-bit
integers via ``decimal`` with very high precision plus an exact cross-product
tie-break written from scratch.  It exists so tests can compare a production
result with a second opinion that does not share the fast-path code.

Nothing in the production path imports this module.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal, localcontext
from functools import cmp_to_key
from typing import Dict, List, Sequence, Tuple


def _seed_row_hash(seed: bytes, row_id: str) -> int:
    rid = row_id.encode("utf-8")
    data = seed + len(rid).to_bytes(8, "big") + rid
    return int.from_bytes(hashlib.sha256(data).digest(), "big") + 1


def brute_rank(seed: bytes, rows: Sequence[dict]) -> List[dict]:
    """Rank rows using 200-digit Decimal priorities (independent reference).

    ``rows`` are dicts with ``row_id``, ``group``, ``weight``.  Returns dicts
    with ``row_id``, ``weight``, ``key`` (int) and ``p`` (Decimal priority),
    sorted by the exact rule: priority desc, row_id code-point asc.
    """
    D = Decimal(2) ** 256 + 1
    with localcontext() as ctx:
        ctx.prec = 200
        out = []
        for r in rows:
            k = _seed_row_hash(seed, r["row_id"])
            u = Decimal(k) / D
            p = u ** (Decimal(1) / Decimal(r["weight"]))
            out.append({"row_id": r["row_id"], "group": r["group"],
                        "weight": r["weight"], "key": k, "p": p})
    # ``p`` at 200 digits is exact enough to separate all but physically
    # impossible ties; exact ties fall back to code-point ordering.
    out.sort(key=lambda x: (-x["p"], x["row_id"]))
    return out


def exact_cross_compare(a1: int, w1: int, a2: int, w2: int) -> int:
    """Independent exact comparator: compare a1^(1/w1) vs a2^(1/w2).

    Uses the integer cross product directly (written without the production
    helper), returning -1/0/+1.
    """
    D = (1 << 256) + 1
    left = pow(a1, w2) * pow(D, w1)
    right = pow(a2, w1) * pow(D, w2)
    if left == right:
        return 0
    return 1 if left > right else -1


def select_reference(seed: bytes, rows: Sequence[dict], quotas: Dict[str, int]
                     ) -> Dict[str, List[str]]:
    """Reference per-group selection for a small row set."""
    from collections import defaultdict
    by_group: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    result: Dict[str, List[str]] = {}
    for g in sorted(by_group):
        ranked = brute_rank(seed, by_group[g])
        q = max(0, int(quotas.get(g, 0)))
        result[g] = [r["row_id"] for r in ranked[:q]]
    return result
