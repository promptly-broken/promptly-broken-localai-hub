"""
prm_scratch.py — Oracle-computed scratchpads for PRM v3.

For a remaining state of 2 or 3 numbers, the scratchpad lists the exact pairwise
results and any move that reaches 24, so the final <verdict> token becomes a
*reading* task (copy "hit"/"none") instead of an implicit search.

Single source of truth for training (generate_prm_dataset_v3.py) and, later, inference.
Scratchpad is emitted BEFORE the verdict only when the resulting state has 2 or 3 numbers.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

TARGET = Fraction(24)
SCRATCH_OPEN = "<scratch>"
SCRATCH_CLOSE = "</scratch>"
SCRATCH_SIZES = (2, 3)


def _n(x: Fraction) -> str:
    """Operand text; negatives parenthesised so expressions stay unambiguous."""
    return f"({x})" if x < 0 else str(x)


def pair_results(a: Fraction, b: Fraction) -> list[tuple[Fraction, str]]:
    """All distinct values obtainable from a,b with + - * / (each order), with an expression."""
    cand = [
        (a + b, f"{_n(a)}+{_n(b)}"),
        (a - b, f"{_n(a)}-{_n(b)}"),
        (b - a, f"{_n(b)}-{_n(a)}"),
        (a * b, f"{_n(a)}*{_n(b)}"),
    ]
    if b != 0:
        cand.append((a / b, f"{_n(a)}/{_n(b)}"))
    if a != 0:
        cand.append((b / a, f"{_n(b)}/{_n(a)}"))
    seen: set[Fraction] = set()
    out: list[tuple[Fraction, str]] = []
    for v, e in cand:
        if v not in seen:
            seen.add(v)
            out.append((v, e))
    return out


def _hit_for_pair(a: Fraction, b: Fraction, z: Fraction | None) -> str:
    """'hit <expr chain>' if this pair (+ optional third number z) can reach 24, else 'none'."""
    results = pair_results(a, b)
    if z is None:  # 2-number state: the pair result itself must be 24
        for v, e in results:
            if v == TARGET:
                return f"hit {e}=24"
        return "none"
    for v, e in results:
        for v2, e2 in pair_results(v, z):
            if v2 == TARGET:
                return f"hit {e}={v}, {e2}=24"
    return "none"


def build_scratchpad(state: Sequence[Fraction]) -> tuple[str, bool]:
    """Return (scratchpad_text, reachable) for a 2- or 3-number state."""
    s = sorted(Fraction(x) for x in state)
    if len(s) == 2:
        a, b = s
        vals = ", ".join(str(v) for v, _ in pair_results(a, b))
        hit = _hit_for_pair(a, b, None)
        text = f"{SCRATCH_OPEN}\n[{a}, {b}]: {vals}\n{hit}\n{SCRATCH_CLOSE}"
        return text, hit != "none"
    if len(s) == 3:
        a, b, c = s
        lines = [f"[{a}, {b}, {c}]"]
        reachable = False
        for (x, y), z in (((a, b), c), ((a, c), b), ((b, c), a)):
            vals = ", ".join(str(v) for v, _ in pair_results(x, y))
            hit = _hit_for_pair(x, y, z)
            reachable = reachable or hit != "none"
            lines.append(f"{x},{y} | {z}: {vals}; {hit}")
        lines.append(f"reachable: {'yes' if reachable else 'no'}")
        return f"{SCRATCH_OPEN}\n" + "\n".join(lines) + f"\n{SCRATCH_CLOSE}", reachable
    raise ValueError(f"scratchpad only defined for sizes {SCRATCH_SIZES}, got {len(s)}")


def needs_scratchpad(n_remaining: int) -> bool:
    return n_remaining in SCRATCH_SIZES
