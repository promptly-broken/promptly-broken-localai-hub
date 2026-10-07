#!/usr/bin/env python3
"""
generate_prm_dataset_v3.py — Adds oracle scratchpads to the v2 dataset (fallback P7 rung 3).

Reads prm_data_v2/{train,valid,test,bench_states}.jsonl and rewrites completions for rows whose
resulting state has 2 or 3 numbers:  <scratch>...</scratch>\n<verdict ...>...  (verdict still scored
from its logit). Splits are inherited unchanged, so v3 test results are directly comparable to v2.

arithmetic_error rows with 2/3 numbers are dropped (the search only proposes legal moves, and their
n_numbers refers to the pre-step state, which would make the format rule ambiguous).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prm_scratch import build_scratchpad, needs_scratchpad

SPLITS = ["train", "valid", "test", "bench_states"]


def convert_row(row: dict) -> tuple[dict | None, str]:
    """Returns (new_row_or_None, status) where status in {unchanged, scratch, dropped_arith, mismatch}."""
    n = int(row["n_numbers"])
    if not needs_scratchpad(n):
        return row, "unchanged"
    if row.get("error_type") == "arithmetic_error":
        return None, "dropped_arith"
    state = [Fraction(x) for x in row["state"]]
    if len(state) != n:
        return None, "mismatch"
    text, reachable = build_scratchpad(state)
    if reachable != (row["label"] == 1.0):
        return None, "mismatch"  # oracle disagreement; never train on it
    new = dict(row)
    new["completion"] = text + "\n" + row["completion"]
    new["has_scratch"] = True
    return new, "scratch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="prm_data_v2")
    ap.add_argument("--out-dir", default="prm_data_v3")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta: dict = {"source": str(in_dir), "splits": {}}
    for split in SPLITS:
        src = in_dir / f"{split}.jsonl"
        if not src.exists():
            continue
        stats: Counter = Counter()
        max_chars = 0
        labels: Counter = Counter()
        out_rows = []
        with open(src, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                new, status = convert_row(row)
                stats[status] += 1
                if new is None:
                    continue
                out_rows.append(new)
                labels[new["label"]] += 1
                max_chars = max(max_chars, len(new["prompt"]) + len(new["completion"]))
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as g:
            for r in out_rows:
                g.write(json.dumps(r) + "\n")
        meta["splits"][split] = {
            "rows": len(out_rows),
            "status": dict(stats),
            "label_counts": {str(k): v for k, v in labels.items()},
            "max_chars_prompt_plus_completion": max_chars,
        }
        print(f"{split:13s} rows={len(out_rows):5d} status={dict(stats)} labels={dict(labels)} max_chars={max_chars}")

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved dataset to {out_dir}/")


if __name__ == "__main__":
    main()
