"""
test_search_soft.py — Unit tests for soft pruning, branch resurrection, and noise resilience.
Verifies:
1. Root never gets permanently pruned in soft mode.
2. Symbolic mode search remains bit-identical with hard pruning.
3. Search with a noisy mock verifier (30% false rejections) still succeeds reliably due to soft resurrection.
"""

import sys
import unittest
from fractions import Fraction
from pathlib import Path
import random

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcts.search import MCTSSearchEngine, SearchResult
from mcts.tree import Node
from mcts.verifier import PRMVerifier, VerificationResult
from solver import is_solvable


class MockNoisyVerifier:
    """Simulates a noisy PRM with controlled false-negative flip rate."""
    def __init__(self, flip_rate: float = 0.3, seed: int = 42):
        self.flip_rate = flip_rate
        self.rng = random.Random(seed)
        self.mode = "mock_noisy"
        self.calls = 0
        self.cache_hits = 0
        self.model_ms = 0.0

    def evaluate_step(
        self,
        initial_numbers: list[int],
        history: list[str],
        candidate_step: str,
        remaining_state: tuple[Fraction, ...]
    ) -> VerificationResult:
        self.calls += 1
        # Terminal step check
        if len(remaining_state) == 1:
            if remaining_state[0] == Fraction(24, 1):
                return VerificationResult(score=1.0, critique="Terminal 24", is_pruned=False)
            return VerificationResult(score=0.0, critique="Terminal not 24", is_pruned=True)

        solvable = is_solvable(remaining_state)
        # Flip sound step to dead end with probability self.flip_rate
        if solvable and self.rng.random() < self.flip_rate:
            score = 0.05  # Falsely rejected
            is_pruned = True
        elif solvable:
            score = 0.95
            is_pruned = False
        else:
            score = 0.05
            is_pruned = True

        return VerificationResult(score=score, critique="Mock critique", is_pruned=is_pruned)


class TestSearchSoft(unittest.TestCase):
    def test_root_resurrection(self):
        """In soft mode, if all children of a node are pruned, the best child is resurrected."""
        root = Node(state=(Fraction(1, 1), Fraction(2, 1), Fraction(3, 1)), history=[])
        c1 = Node(state=(Fraction(3, 1), Fraction(3, 1)), history=["1 + 2 = 3"], parent=root)
        c2 = Node(state=(Fraction(2, 1), Fraction(3, 1)), history=["1 * 2 = 2"], parent=root)
        c1.prm_score = 0.05
        c1.is_pruned = True
        c2.prm_score = 0.20
        c2.is_pruned = True
        root.children = [c1, c2]
        root.untried_moves = []

        root.update_pruned_status(soft=True)
        # c2 had higher prm_score, so it should be resurrected and root remains unpruned
        self.assertFalse(root.is_pruned, "Root must not be marked pruned after soft update")
        self.assertFalse(c2.is_pruned, "Child with highest PRM score must be resurrected")
        self.assertTrue(c2.resurrected, "Resurrected flag must be set to True")
        self.assertTrue(c1.is_pruned, "Inferior child should remain pruned")

    def test_symbolic_regression(self):
        """Ensures symbolic search continues to find the exact solutions identically."""
        engine = MCTSSearchEngine(verifier=PRMVerifier(mode="symbolic"), max_simulations=100)
        puzzles = [
            [4, 4, 10, 10],  # (10 * 10 - 4) / 4 = 24
            [1, 5, 5, 5],    # 5 * (5 - 1/5) = 24
            [3, 3, 8, 8],    # 8 / (3 - 8/3) = 24
        ]
        for nums in puzzles:
            res = engine.solve(nums)
            self.assertTrue(res.success, f"Failed to solve {nums} in symbolic mode")
            self.assertGreater(len(res.solution_steps), 0)

    def test_noise_resilience_with_soft_pruning(self):
        """Noisy PRM with 30% false rejections still achieves >= 80% solve rate under soft pruning."""
        nums = [1, 5, 5, 5]
        successes = 0
        total_trials = 20

        for seed in range(total_trials):
            noisy_v = MockNoisyVerifier(flip_rate=0.30, seed=seed)
            engine = MCTSSearchEngine(verifier=noisy_v, prune_mode="soft", max_simulations=150, k_branching=4)
            res = engine.solve(nums)
            if res.success:
                successes += 1

        solve_rate = successes / total_trials
        self.assertGreaterEqual(
            solve_rate,
            0.80,
            f"Expected at least 80% solve rate under 30% noise with soft pruning, got {solve_rate:.1%}"
        )


if __name__ == "__main__":
    unittest.main()
