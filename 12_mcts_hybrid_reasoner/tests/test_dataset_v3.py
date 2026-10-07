"""
test_dataset_v3.py — Verifies scratchpad correctness and v3 dataset integrity (unittest only).
"""

import json
import sys
import unittest
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prm_scratch import build_scratchpad
from solver import is_solvable

DATA_DIR = Path(__file__).resolve().parent.parent / "prm_data_v3"


def F(*xs):
    return [Fraction(x) for x in xs]


class TestScratchpadUnits(unittest.TestCase):
    def test_size2_hit_and_none(self):
        text, ok = build_scratchpad(F(3, 8))
        self.assertTrue(ok)
        self.assertIn("hit 3*8=24", text)
        text, ok = build_scratchpad(F(1, 1))
        self.assertFalse(ok)
        self.assertIn("\nnone\n", text)

    def test_size3_matches_solver_exhaustive_sample(self):
        import itertools
        for trio in itertools.combinations_with_replacement(range(1, 8), 3):
            _, ok = build_scratchpad(F(*trio))
            self.assertEqual(ok, is_solvable(tuple(F(*trio))), f"mismatch for {trio}")

    def test_size3_known_cases(self):
        _, ok = build_scratchpad(F(2, 4, 3))      # 2*4=8, 8*3=24
        self.assertTrue(ok)
        _, ok = build_scratchpad(F(1, 1, 1))
        self.assertFalse(ok)

    def test_fraction_and_negative(self):
        text, ok = build_scratchpad([Fraction(1, 3), Fraction(8)])
        self.assertTrue(ok)  # 1/3 reciprocal 3 * 8 = 24
        self.assertIn("hit", text)


@unittest.skipUnless((DATA_DIR / "train.jsonl").exists(), "prm_data_v3 not generated yet")
class TestDatasetV3(unittest.TestCase):
    def rows(self, name):
        with open(DATA_DIR / f"{name}.jsonl", encoding="utf-8") as f:
            return [json.loads(l) for l in f]

    def test_every_scratch_row_consistent(self):
        for split in ("train", "valid", "test", "bench_states"):
            for r in self.rows(split):
                comp = r["completion"]
                n = r["n_numbers"]
                if n in (2, 3):
                    self.assertTrue(comp.startswith("<scratch>"), (split, r["state"]))
                    self.assertIn("</scratch>\n<verdict score=", comp)
                    reach = "reachable: yes" in comp if n == 3 else "\nnone\n" not in comp.split("</scratch>")[0]
                    self.assertEqual(reach, r["label"] == 1.0, (split, r["state"]))
                else:
                    self.assertTrue(comp.startswith("<verdict score="), (split, n))

    def test_no_arithmetic_error_small_states(self):
        for split in ("train", "valid", "test", "bench_states"):
            for r in self.rows(split):
                if r["n_numbers"] in (2, 3):
                    self.assertNotEqual(r.get("error_type"), "arithmetic_error")

    def test_splits_disjoint_on_state(self):
        keys = {s: {",".join(r["state"]) for r in self.rows(s)} for s in ("train", "valid", "test", "bench_states")}
        self.assertFalse(keys["train"] & keys["test"])
        self.assertFalse(keys["train"] & keys["valid"])
        self.assertFalse(keys["train"] & keys["bench_states"])

    def test_class_balance_train(self):
        rows = self.rows("train")
        pos = sum(1 for r in rows if r["label"] == 1.0) / len(rows)
        self.assertTrue(0.45 <= pos <= 0.55, pos)


if __name__ == "__main__":
    unittest.main()