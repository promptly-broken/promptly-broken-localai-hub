"""
Unit tests for the MCTS Search Engine
"""

import unittest
from mcts_reasoner.mcts.search import MCTSSearchEngine
from mcts_reasoner.mcts.verifier import PRMVerifier


class TestMCTSSearch(unittest.TestCase):

    def test_solve_deceptive_puzzle_3388(self):
        """Test solving [3, 3, 8, 8] which requires fractional division."""
        engine = MCTSSearchEngine(
            verifier=PRMVerifier(mode="symbolic"),
            max_simulations=40,
            k_branching=5
        )
        res = engine.solve([3, 3, 8, 8])

        self.assertTrue(res.success, f"Failed to solve [3, 3, 8, 8]: {res.verification_message}")
        self.assertEqual(len(res.solution_steps), 3)
        self.assertGreater(res.pruned_nodes, 0)
        print(f"\n[3, 3, 8, 8] Solved in {res.execution_time_sec*1000:.2f}ms! Pruned {res.pruned_nodes} dead-end nodes.")
        for s in res.solution_steps:
            print(f"  {s}")

    def test_solve_classic_4468(self):
        """Test solving [4, 4, 6, 8]."""
        engine = MCTSSearchEngine(
            verifier=PRMVerifier(mode="symbolic"),
            max_simulations=20,
            k_branching=4
        )
        res = engine.solve([4, 4, 6, 8])
        self.assertTrue(res.success)
        self.assertEqual(len(res.solution_steps), 3)


if __name__ == "__main__":
    unittest.main()
