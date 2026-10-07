"""
test_dataset_v2.py — Verification suite for PRM v2 oracle-labelled dataset.
Uses standard unittest without external pytest dependencies.
Ensures zero data leakage, correct labels, format compliance, and class balance.
"""

import json
import sys
import unittest
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver import is_solvable

DATA_DIR = Path(__file__).resolve().parent.parent / "prm_data_v2"


class TestDatasetV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train_f = DATA_DIR / "train.jsonl"
        valid_f = DATA_DIR / "valid.jsonl"
        test_f = DATA_DIR / "test.jsonl"
        bench_f = DATA_DIR / "bench_states.jsonl"

        if not (train_f.exists() and valid_f.exists() and test_f.exists() and bench_f.exists()):
            raise unittest.SkipTest("prm_data_v2 files not found; run generate_prm_dataset_v2.py first")

        def read_jsonl(p):
            with open(p, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]

        cls.train = read_jsonl(train_f)
        cls.valid = read_jsonl(valid_f)
        cls.test = read_jsonl(test_f)
        cls.bench = read_jsonl(bench_f)

    def test_leakage_and_splits_disjoint(self):
        """Verifies that no resulting-state string is shared between train and bench, or across splits."""
        def get_state_keys(records):
            return set(",".join(r["state"]) for r in records)

        train_states = get_state_keys(self.train)
        valid_states = get_state_keys(self.valid)
        test_states = get_state_keys(self.test)
        bench_states = get_state_keys(self.bench)

        # Disjointness checks
        self.assertEqual(len(train_states.intersection(bench_states)), 0, "Train states overlap with benchmark states!")
        self.assertEqual(len(train_states.intersection(valid_states)), 0, "Train states overlap with validation states!")
        self.assertEqual(len(train_states.intersection(test_states)), 0, "Train states overlap with test states!")
        self.assertEqual(len(valid_states.intersection(test_states)), 0, "Valid states overlap with test states!")

    def test_no_benchmark_source_puzzles(self):
        """Ensures no puzzle_id with ranks 901-950 appears in train."""
        for r in self.train:
            p_id = r.get("puzzle_id", "")
            if p_id.startswith("tot_"):
                rank = int(p_id.split("_")[1])
                self.assertFalse(901 <= rank <= 950, f"Benchmark puzzle rank {rank} found in train!")

    def test_train_class_balance(self):
        """Checks that train set is balanced between 48% and 52%."""
        pos_count = sum(1 for r in self.train if r["label"] == 1.0)
        ratio = pos_count / len(self.train)
        self.assertTrue(0.48 <= ratio <= 0.52, f"Train class balance out of range: {ratio:.2%}")

    def test_format_and_completions(self):
        """Validates completion structure: <verdict> first, then <critique>, ending with <|im_end|>."""
        for split_name, records in [("train", self.train), ("valid", self.valid), ("bench", self.bench)]:
            sample = records[:200]
            for r in sample:
                comp = r["completion"]
                self.assertTrue(
                    comp.startswith('<verdict score="1.0">ACCEPT</verdict>')
                    or comp.startswith('<verdict score="0.0">REJECT</verdict>'),
                    f"Invalid verdict header: {comp[:50]}",
                )
                self.assertIn("<critique>", comp, "Missing <critique> tag")
                self.assertIn("</critique>", comp, "Missing </critique> tag")
                self.assertTrue(comp.endswith("<|im_end|>"), "Completion does not end with <|im_end|>")
                self.assertTrue(r["prompt"].startswith("<|im_start|>user\n"), "Prompt missing user turn tag")
                self.assertTrue(r["prompt"].endswith("<|im_start|>assistant\n"), "Prompt missing assistant turn tag")

    def test_label_soundness_against_oracle(self):
        """Verifies that non-error rows strictly match is_solvable(state)."""
        sample = self.train[:500] + self.valid[:200]
        for r in sample:
            state_fracs = tuple(Fraction(x) for x in r["state"])
            err = r.get("error_type")
            if err in ["arithmetic_error", "terminal_deviation"]:
                self.assertEqual(r["label"], 0.0, f"Error row must have label 0.0: {r}")
            else:
                expected_solvable = is_solvable(state_fracs)
                expected_label = 1.0 if expected_solvable else 0.0
                self.assertEqual(
                    r["label"],
                    expected_label,
                    f"Mismatch with oracle for state {state_fracs}: got {r['label']} vs {expected_label}",
                )


if __name__ == "__main__":
    unittest.main()