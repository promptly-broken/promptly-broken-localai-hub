"""
Candidate Step Generator for Game of 24
Interfaces with local LLM (Ollama / MLX) to generate single-step candidates,
with fallback to programmatic expansion for testing.
"""

from __future__ import annotations
import json
import re
import urllib.request
from fractions import Fraction
from typing import NamedTuple

from mcts_reasoner.solver import get_possible_moves, Step


class CandidateMove(NamedTuple):
    step_str: str
    remaining: tuple[Fraction, ...]
    prior_prob: float


class StepGenerator:
    """Generates K candidate moves for the next step in the reasoning chain."""

    def __init__(
        self,
        model_name: str = "qwen2.5-coder:7b-instruct",
        ollama_url: str = "http://127.0.0.1:11434/api/generate",
        temperature: float = 0.7
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.temperature = temperature

    def pop_untried_candidates(
        self,
        node,
        k: int = 3
    ) -> list[CandidateMove]:
        """
        Pops up to k untried legal moves directly from the node's untried_moves list.
        """
        candidates: list[CandidateMove] = []
        num_to_pop = min(k, len(node.untried_moves))
        if num_to_pop == 0:
            return []

        prior = 1.0 / num_to_pop
        for _ in range(num_to_pop):
            move = node.untried_moves.pop(0)
            candidates.append(CandidateMove(
                step_str=move.description(),
                remaining=move.remaining,
                prior_prob=prior
            ))
        return candidates

    def generate_candidates_llm(
        self,
        initial_numbers: list[int],
        history: list[str],
        current_state: tuple[Fraction, ...],
        k: int = 3
    ) -> list[CandidateMove]:
        """
        Calls local LLM to propose K alternative next steps.
        """
        step_num = len(history) + 1
        rem_str = [str(x) for x in current_state]
        history_text = "\n".join(history) if history else "None (Start of puzzle)"

        prompt = (
            f"You are solving the Game of 24.\n"
            f"Initial numbers: {initial_numbers}\n"
            f"Previous steps:\n{history_text}\n"
            f"Remaining available numbers: {rem_str}\n\n"
            f"Propose up to {k} distinct possible candidate operations for Step {step_num}.\n"
            f"Each candidate must pick two numbers from {rem_str} and an operation (+, -, *, /).\n"
            f"Format strictly as:\n"
            f"Candidate 1: a op b = c\n"
            f"Candidate 2: a op b = c\n"
            f"Candidate 3: a op b = c\n"
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 150
            }
        }

        try:
            req = urllib.request.Request(
                self.ollama_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                response_text = data.get("response", "")

            candidates = self._parse_llm_candidates(response_text, current_state)
            if candidates:
                return candidates[:k]
        except Exception:
            pass

        # Fallback to possible moves
        possible = get_possible_moves(current_state)
        candidates = []
        prior = 1.0 / max(1, min(k, len(possible)))
        for m in possible[:k]:
            candidates.append(CandidateMove(
                step_str=m.description(),
                remaining=m.remaining,
                prior_prob=prior
            ))
        return candidates

    def _parse_llm_candidates(
        self,
        text: str,
        current_state: tuple[Fraction, ...]
    ) -> list[CandidateMove]:
        """Extracts valid candidate steps from LLM text."""
        candidates = []
        lines = text.strip().split("\n")
        
        for line in lines:
            clean_line = re.sub(r"^(Candidate\s*\d+:|\d+[\.\)]|\-\s*)", "", line.strip()).strip()
            if not clean_line or "=" not in clean_line:
                continue

            left, right = clean_line.split("=", 1)
            left = left.strip()
            right = right.strip()

            found_op = None
            for op in [" + ", " - ", " * ", " / "]:
                if op in left:
                    parts = left.split(op, 1)
                    found_op = op.strip()
                    n1_str, n2_str = parts[0].strip(), parts[1].strip()
                    break

            if not found_op:
                continue

            try:
                n1 = Fraction(n1_str)
                n2 = Fraction(n2_str)
                res = Fraction(right.split()[0].strip())
            except ValueError:
                continue

            state_list = list(current_state)
            if n1 in state_list:
                state_list.remove(n1)
                if n2 in state_list:
                    state_list.remove(n2)
                    state_list.append(res)
                    new_state = tuple(sorted(state_list))
                    step_desc = f"{n1} {found_op} {n2} = {res} (Remaining: {[str(x) for x in new_state]})"
                    candidates.append(CandidateMove(
                        step_str=step_desc,
                        remaining=new_state,
                        prior_prob=0.8
                    ))

        return candidates
