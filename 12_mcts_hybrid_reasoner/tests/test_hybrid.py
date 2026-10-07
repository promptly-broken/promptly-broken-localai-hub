"""
test_hybrid.py — Unit tests verifying Hybrid Verifier routing and MCTS integration.
Uses unittest and mock without loading heavy model weights.
"""

import sys
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcts.verifier import PRMVerifier, VerificationResult
from mcts.search import MCTSSearchEngine


class TestHybridVerifier(unittest.TestCase):
    def test_routing_by_state_size(self):
        verifier = PRMVerifier(mode="hybrid")

        # Mock both _evaluate_symbolic and _evaluate_mlx_logit
        dummy_sym = VerificationResult(score=1.0, critique="sym", is_pruned=False)
        dummy_mlx = VerificationResult(score=0.85, critique="mlx", is_pruned=False)

        with mock.patch.object(verifier, "_evaluate_symbolic", return_value=dummy_sym) as mock_sym, \
             mock.patch.object(verifier, "_evaluate_mlx_logit", return_value=dummy_mlx) as mock_mlx:

            # Size 1: terminal shortcut (exact check)
            res1 = verifier.evaluate_step([1, 2, 3, 4], [], "Step 3: 20 + 4 = 24", (Fraction(24),))
            self.assertEqual(res1.score, 1.0)
            mock_sym.assert_not_called()
            mock_mlx.assert_not_called()

            # Size 2: pre-terminal -> routes to symbolic
            res2 = verifier.evaluate_step([1, 2, 3, 4], [], "Step 2: 2 * 3 = 6", (Fraction(6), Fraction(4)))
            self.assertEqual(res2.score, 1.0)
            mock_sym.assert_called_once()
            mock_mlx.assert_not_called()

            # Size 3: intermediate search -> routes to neural MLX PRM
            res3 = verifier.evaluate_step([1, 2, 3, 4], [], "Step 1: 1 + 2 = 3", (Fraction(3), Fraction(3), Fraction(4)))
            self.assertEqual(res3.score, 0.85)
            mock_mlx.assert_called_once()

    def test_search_engine_soft_pruning_default(self):
        verifier = PRMVerifier(mode="hybrid")
        engine = MCTSSearchEngine(verifier=verifier)
        self.assertEqual(engine.prune_mode, "soft")


if __name__ == "__main__":
    unittest.main()
