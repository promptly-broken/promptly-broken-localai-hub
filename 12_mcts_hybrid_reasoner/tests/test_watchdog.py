#!/usr/bin/env python3
"""
test_watchdog.py — Verification suite for NaN watchdog in train_prm.py
Uses standard unittest without external dependencies.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_prm import run_training_with_watchdog, NAN_PATTERN


class TestWatchdog(unittest.TestCase):
    def test_regex_patterns(self):
        # Should detect
        self.assertTrue(NAN_PATTERN.search("Iter 340: loss nan, lr 1.000e-04"))
        self.assertTrue(NAN_PATTERN.search("Iter 400: Val loss nan, Val took 28s"))
        self.assertTrue(NAN_PATTERN.search("loss: nan"))
        self.assertTrue(NAN_PATTERN.search("loss = nan"))
        self.assertTrue(NAN_PATTERN.search("loss: inf"))
        self.assertTrue(NAN_PATTERN.search("loss: -inf"))
        self.assertTrue(NAN_PATTERN.search("{'loss': 'nan'}"))
        self.assertTrue(NAN_PATTERN.search('{"loss": "nan"}'))

        # Should NOT detect
        self.assertFalse(NAN_PATTERN.search("Iter 200: loss 0.951, lr 1.000e-04"))
        self.assertFalse(NAN_PATTERN.search("Iter 400: Val loss 0.687, Val took 28s"))
        self.assertFalse(NAN_PATTERN.search("loss: 0.123"))
        self.assertFalse(NAN_PATTERN.search("Saved final weights to adapters.safetensors"))

    def test_watchdog_detects_nan_and_aborts(self):
        # Subprocess command that outputs nan
        cmd = [
            sys.executable,
            "-c",
            "import sys, time; sys.stdout.write('Iter 20: loss 0.812\\n'); sys.stdout.flush(); time.sleep(0.05); sys.stdout.write('Iter 40: loss nan\\n'); sys.stdout.flush(); time.sleep(5.0)"
        ]

        with self.assertRaises(SystemExit) as cm:
            run_training_with_watchdog(cmd, watchdog_enabled=True)

        self.assertEqual(cm.exception.code, 2)

    def test_watchdog_passes_valid_run(self):
        cmd = [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('Iter 20: loss 0.812\\nIter 40: loss 0.541\\n'); sys.stdout.flush()"
        ]

        ret = run_training_with_watchdog(cmd, watchdog_enabled=True)
        self.assertEqual(ret, 0)


if __name__ == "__main__":
    unittest.main()
