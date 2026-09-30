"""
Exact State-Space Solver & Deterministic Verifier for Game of 24
Using rational arithmetic (fractions.Fraction) to prevent floating-point inaccuracies.
"""

from __future__ import annotations
from fractions import Fraction
from itertools import combinations
from typing import NamedTuple


class Step(NamedTuple):
    num1: Fraction
    num2: Fraction
    op: str
    result: Fraction
    remaining: tuple[Fraction, ...]

    def to_string(self) -> str:
        return f"{self.num1} {self.op} {self.num2} = {self.result}"

    def description(self) -> str:
        rem_str = [str(x) for x in self.remaining]
        return f"{self.num1} {self.op} {self.num2} = {self.result} (Remaining: {rem_str})"


def get_possible_moves(numbers: tuple[Fraction, ...]) -> list[Step]:
    """
    Given a tuple of Fractions, return all valid pairwise operations.
    Handles commutativity and division-by-zero checks.
    """
    moves = []
    n = len(numbers)
    if n < 2:
        return []

    for i in range(n):
        for j in range(i + 1, n):
            a = numbers[i]
            b = numbers[j]
            rest = tuple(numbers[k] for k in range(n) if k != i and k != j)

            # 1. Addition (commutative: a + b)
            moves.append(Step(a, b, "+", a + b, tuple(sorted(rest + (a + b,)))))

            # 2. Multiplication (commutative: a * b)
            moves.append(Step(a, b, "*", a * b, tuple(sorted(rest + (a * b,)))))

            # 3. Subtraction (non-commutative: a - b, b - a)
            moves.append(Step(a, b, "-", a - b, tuple(sorted(rest + (a - b,)))))
            moves.append(Step(b, a, "-", b - a, tuple(sorted(rest + (b - a,)))))

            # 4. Division (non-commutative, no division by zero)
            if b != 0:
                moves.append(Step(a, b, "/", a / b, tuple(sorted(rest + (a / b,)))))
            if a != 0:
                moves.append(Step(b, a, "/", b / a, tuple(sorted(rest + (b / a,)))))

    # Deduplicate moves by (operation_string, remaining_numbers)
    unique_moves: list[Step] = []
    seen = set()
    for m in moves:
        key = (m.to_string(), m.remaining)
        if key not in seen:
            seen.add(key)
            unique_moves.append(m)

    return unique_moves


def is_solvable(numbers: tuple[Fraction, ...], target: Fraction = Fraction(24, 1)) -> bool:
    """
    Recursively checks if a multiset of numbers can reach target.
    Returns True if at least one winning path exists, False otherwise.
    """
    if len(numbers) == 1:
        return numbers[0] == target

    for move in get_possible_moves(numbers):
        if is_solvable(move.remaining, target):
            return True

    return False


def find_winning_paths(
    numbers: tuple[Fraction, ...],
    target: Fraction = Fraction(24, 1),
    max_paths: int = 10
) -> list[list[Step]]:
    """
    Returns up to max_paths complete solution paths (sequences of Steps)
    reaching the target.
    """
    solutions: list[list[Step]] = []

    def dfs(current_nums: tuple[Fraction, ...], history: list[Step]):
        if len(current_nums) == 1:
            if current_nums[0] == target:
                solutions.append(history)
            return

        if len(solutions) >= max_paths:
            return

        for move in get_possible_moves(current_nums):
            dfs(move.remaining, history + [move])
            if len(solutions) >= max_paths:
                return

    dfs(numbers, [])
    return solutions


def classify_next_steps(
    numbers: tuple[Fraction, ...],
    target: Fraction = Fraction(24, 1)
) -> tuple[list[Step], list[Step]]:
    """
    Partitions all candidate next steps from the current state into:
      - winning_steps: moves that preserve solvability (Label = 1.0)
      - dead_end_steps: moves that doom the branch to an unsolvable state (Label = 0.0)
    """
    winning: list[Step] = []
    dead_ends: list[Step] = []

    all_moves = get_possible_moves(numbers)
    for move in all_moves:
        if is_solvable(move.remaining, target):
            winning.append(move)
        else:
            dead_ends.append(move)

    return winning, dead_ends


def verify_solution(
    initial_numbers: list[int],
    steps: list[str],
    target: Fraction = Fraction(24, 1)
) -> tuple[bool, str]:
    """
    Deterministically verifies an LLM-generated solution trajectory:
    1. Parses each step 'num1 op num2 = res'
    2. Checks if num1 and num2 exist in the available pool
    3. Verifies arithmetic correctness (no hallucinated sums/products)
    4. Confirms final number equals target and all numbers were properly consumed
    """
    current_pool = [Fraction(x, 1) for x in initial_numbers]

    for idx, raw_step in enumerate(steps, 1):
        # Clean line
        cleaned = raw_step.strip()
        if not cleaned:
            continue

        # Expect format 'a op b = c' (possibly followed by remaining numbers in parentheses)
        main_part = cleaned.split("(")[0].strip()
        if "=" not in main_part:
            return False, f"Step {idx} invalid format: missing '=' in '{cleaned}'"

        left, right = main_part.split("=")
        left = left.strip()
        right = right.strip()

        # Parse operator with spaces around it: " + ", " - ", " * ", " / "
        found_op = None
        for op in [" + ", " - ", " * ", " / "]:
            if op in left:
                parts = left.split(op, 1)
                found_op = op.strip()
                num1_str, num2_str = parts[0].strip(), parts[1].strip()
                break

        # Fallback if no spaces around operator
        if not found_op:
            for op in ["+", "-", "*", "/"]:
                # If operator is '/', ensure it's not just inside a fraction like '1/3'
                if op == "/":
                    # find the division operator that separates two terms
                    idx_slash = left.find(" / ")
                    if idx_slash != -1:
                        found_op = "/"
                        num1_str = left[:idx_slash].strip()
                        num2_str = left[idx_slash + 3:].strip()
                        break
                elif op in left:
                    parts = left.split(op, 1)
                    found_op = op
                    num1_str, num2_str = parts[0].strip(), parts[1].strip()
                    break

        if not found_op:
            return False, f"Step {idx} invalid operator in '{left}'"

        try:
            n1 = Fraction(num1_str)
            n2 = Fraction(num2_str)
            res = Fraction(right)
        except ValueError as e:
            return False, f"Step {idx} numerical parsing error: {e}"

        # Verify n1 and n2 are in current_pool
        if n1 not in current_pool:
            return False, f"Step {idx} hallucinated operand: {n1} not in available numbers {[str(x) for x in current_pool]}"
        current_pool.remove(n1)

        if n2 not in current_pool:
            return False, f"Step {idx} hallucinated operand: {n2} not in available numbers {[str(x) for x in current_pool]}"
        current_pool.remove(n2)

        # Verify arithmetic
        expected_res: Fraction
        if found_op == "+":
            expected_res = n1 + n2
        elif found_op == "-":
            expected_res = n1 - n2
        elif found_op == "*":
            expected_res = n1 * n2
        elif found_op == "/":
            if n2 == 0:
                return False, f"Step {idx} division by zero"
            expected_res = n1 / n2

        if res != expected_res:
            return False, f"Step {idx} arithmetic blunder: {n1} {found_op} {n2} should be {expected_res}, got {res}"

        # Add result back to pool
        current_pool.append(res)

    if len(current_pool) != 1:
        return False, f"Solution incomplete: remaining pool has {len(current_pool)} numbers {[str(x) for x in current_pool]}, expected 1"

    if current_pool[0] != target:
        return False, f"Final result is {current_pool[0]}, not target {target}"

    return True, f"Verified solution: {current_pool[0]} == {target}"
