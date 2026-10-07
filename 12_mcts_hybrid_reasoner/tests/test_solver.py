"""
Unit tests for mcts_reasoner/solver.py
"""

import unittest
from fractions import Fraction
from solver import (
    is_solvable,
    find_winning_paths,
    classify_next_steps,
    verify_solution,
)


class TestSolver(unittest.TestCase):

    def test_solvable_known_cases(self):
        # [4, 4, 6, 8] is solvable
        nums1 = (Fraction(4), Fraction(4), Fraction(6), Fraction(8))
        self.assertTrue(is_solvable(nums1))

        # [3, 3, 8, 8] is solvable via fractions: 8 / (3 - 8/3) = 24
        nums2 = (Fraction(3), Fraction(3), Fraction(8), Fraction(8))
        self.assertTrue(is_solvable(nums2))

        # [1, 1, 1, 1] is impossible
        nums_impossible = (Fraction(1), Fraction(1), Fraction(1), Fraction(1))
        self.assertFalse(is_solvable(nums_impossible))

    def test_find_winning_paths(self):
        nums = (Fraction(3), Fraction(3), Fraction(8), Fraction(8))
        paths = find_winning_paths(nums)
        self.assertGreater(len(paths), 0)

        # Check the steps of the winning path
        steps_str = [step.to_string() for step in paths[0]]
        self.assertEqual(len(steps_str), 3)

        # Verify using verify_solution
        valid, msg = verify_solution([3, 3, 8, 8], steps_str)
        self.assertTrue(valid, msg)

    def test_classify_next_steps(self):
        # For [3, 3, 8, 8], exactly one first step is winning: 8 / 3
        nums = (Fraction(3), Fraction(3), Fraction(8), Fraction(8))
        winning, dead_ends = classify_next_steps(nums)

        self.assertEqual(len(winning), 1)
        self.assertEqual(winning[0].to_string(), "8 / 3 = 8/3")
        self.assertGreater(len(dead_ends), 10)

    def test_verify_solution_blunders(self):
        # Arithmetic error
        steps_bad_math = [
            "3 * 8 = 25",
            "25 - 3 = 22",
            "22 + 2 = 24"
        ]
        valid, msg = verify_solution([3, 3, 8, 8], steps_bad_math)
        self.assertFalse(valid)
        self.assertIn("arithmetic blunder", msg)

        # Hallucinated number (not in hand)
        steps_hallucinated = [
            "10 - 2 = 8",
            "3 * 8 = 24"
        ]
        valid, msg = verify_solution([3, 3, 8, 8], steps_hallucinated)
        self.assertFalse(valid)
        self.assertIn("hallucinated operand", msg)


if __name__ == "__main__":
    unittest.main()
