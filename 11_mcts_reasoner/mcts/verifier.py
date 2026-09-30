"""
Hybrid Process Reward Model (PRM) Verifier
Evaluates candidate steps and returns calibrated value score + critique.
"""

from __future__ import annotations
import json
import re
import urllib.request
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

from mcts_reasoner.solver import is_solvable

# Module-level cache: model + tokenizer are loaded once and reused across calls.
_MLX_MODEL = None
_MLX_TOKENIZER = None


class VerificationResult(NamedTuple):
    score: float            # Scaled 0.0 to 1.0
    critique: str           # Verbal explanation
    is_pruned: bool         # Whether MCTS should prune this branch


class PRMVerifier:
    """
    Evaluates individual reasoning steps using:
    - Mode A: Exact Symbolic Oracle (ideal for testing MCTS dynamics and speed)
    - Mode B: Local Ollama / MLX PRM
    """

    # Default adapter path: mcts_reasoner/prm_adapters/ relative to repo root.
    # Resolved at class level so it works regardless of CWD.
    _DEFAULT_ADAPTER_PATH: str = str(
        Path(__file__).resolve().parent.parent / "mcts_reasoner" / "prm_adapters"
    )

    def __init__(
        self,
        mode: str = "symbolic",  # "symbolic" | "ollama" | "mlx"
        prune_threshold: float = 0.25,
        model_name: str = "mlx-community/Qwen2.5-Coder-7B-4bit",
        ollama_url: str = "http://127.0.0.1:11434/api/generate",
        adapter_path: str | None = None,
    ):
        self.mode = mode
        self.prune_threshold = prune_threshold
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.adapter_path = adapter_path or self._DEFAULT_ADAPTER_PATH

    def evaluate_step(
        self,
        initial_numbers: list[int],
        history: list[str],
        candidate_step: str,
        remaining_state: tuple[Fraction, ...]
    ) -> VerificationResult:
        """Main entry point to evaluate a candidate step."""
        if self.mode == "symbolic":
            return self._evaluate_symbolic(remaining_state, candidate_step)
        elif self.mode == "ollama":
            return self._evaluate_ollama(initial_numbers, history, candidate_step, remaining_state)
        elif self.mode == "mlx":
            return self._evaluate_mlx(initial_numbers, history, candidate_step, remaining_state)
        else:
            return self._evaluate_symbolic(remaining_state, candidate_step)

    def _evaluate_symbolic(
        self,
        remaining_state: tuple[Fraction, ...],
        candidate_step: str
    ) -> VerificationResult:
        """Ground-truth symbolic verification using the exact DAG solver."""
        solvable = is_solvable(remaining_state)
        rem_str = [str(x) for x in remaining_state]

        if solvable:
            if len(remaining_state) == 1 and remaining_state[0] == Fraction(24, 1):
                critique = f"Solution verified. '{candidate_step}' correctly produces target 24."
                return VerificationResult(score=1.0, critique=critique, is_pruned=False)

            critique = f"Sound intermediate step. State {rem_str} preserves solvability to 24."
            return VerificationResult(score=1.0, critique=critique, is_pruned=False)
        else:
            critique = f"Dead end. State {rem_str} cannot reach 24 under any combination of operations."
            return VerificationResult(score=0.0, critique=critique, is_pruned=True)

    def _evaluate_mlx(
        self,
        initial_numbers: list[int],
        history: list[str],
        candidate_step: str,
        remaining_state: tuple[Fraction, ...]
    ) -> VerificationResult:
        """Evaluates a reasoning step using the locally trained LoRA PRM adapter.

        Builds the exact ChatML prompt used during fine-tuning, runs MLX inference
        with the adapter, and parses the <verdict score="..."> XML tag to extract
        a calibrated score. Falls back to symbolic verification on any error.
        """
        global _MLX_MODEL, _MLX_TOKENIZER

        # Lazy-load model + adapter once; reuse across all subsequent calls.
        if _MLX_MODEL is None or _MLX_TOKENIZER is None:
            try:
                import os
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                from mlx_lm import load
                print(f"[PRM] Loading MLX model '{self.model_name}' with adapter '{self.adapter_path}'...")
                _MLX_MODEL, _MLX_TOKENIZER = load(
                    self.model_name,
                    adapter_path=self.adapter_path,
                )
                print("[PRM] MLX model loaded. Ready for inference.")
            except Exception as exc:
                print(f"[PRM] MLX load failed ({exc}). Falling back to symbolic.")
                return self._evaluate_symbolic(remaining_state, candidate_step)

        # Build the ChatML prompt — identical format to training data.
        # Training data used "Step N: <description>" prefix; replicate that here.
        step_num = len(history) + 1
        prefix_text = "\n".join(history) if history else "Initial State"
        # Avoid double-prefixing if caller already included "Step N:"
        if not re.match(r"^Step\s+\d+:", candidate_step.strip()):
            candidate_step_prefixed = f"Step {step_num}: {candidate_step}"
        else:
            candidate_step_prefixed = candidate_step
        prompt_text = (
            f"Puzzle Numbers: {initial_numbers} -> Target: 24\n"
            f"Previous Steps:\n{prefix_text}\n\n"
            f"Evaluate Proposed Step:\n{candidate_step_prefixed}\n"
            f"Is this step sound?"
        )
        # Wrap in ChatML tokens exactly as seen during training.
        full_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        try:
            from mlx_lm import generate
            response = generate(
                _MLX_MODEL,
                _MLX_TOKENIZER,
                prompt=full_prompt,
                max_tokens=150,
                verbose=False,
            )
        except Exception as exc:
            print(f"[PRM] MLX generate failed ({exc}). Falling back to symbolic.")
            return self._evaluate_symbolic(remaining_state, candidate_step)

        # Parse <verdict score="...">ACCEPT/REJECT</verdict>
        score_match = re.search(r'<verdict score="([\d.]+)">', response)
        verdict_match = re.search(r'<verdict[^>]*>(.*?)</verdict>', response, re.DOTALL)
        critique_match = re.search(r'<critique>(.*?)</critique>', response, re.DOTALL)

        # Determine score
        if score_match:
            score = float(score_match.group(1))
        elif verdict_match and "ACCEPT" in verdict_match.group(1).upper():
            score = 1.0
        elif verdict_match and "REJECT" in verdict_match.group(1).upper():
            score = 0.0
        else:
            # Model produced an unparseable response — fall back to symbolic.
            return self._evaluate_symbolic(remaining_state, candidate_step)

        critique = (
            critique_match.group(1).strip()
            if critique_match
            else response.strip()
        )
        is_pruned = score < self.prune_threshold

        return VerificationResult(score=score, critique=critique, is_pruned=is_pruned)

    def _evaluate_ollama(
        self,
        initial_numbers: list[int],
        history: list[str],
        candidate_step: str,
        remaining_state: tuple[Fraction, ...]
    ) -> VerificationResult:
        """Evaluates step using local LLM prompt."""
        rem_str = [str(x) for x in remaining_state]
        history_text = "\n".join(history) if history else "Initial State"

        prompt = (
            f"You are a strict Process Reward Model (PRM) verifier for the Game of 24.\n"
            f"Puzzle Numbers: {initial_numbers} -> Target: 24\n"
            f"Previous Steps:\n{history_text}\n\n"
            f"Proposed Step:\n{candidate_step}\n"
            f"Remaining numbers after step: {rem_str}\n\n"
            f"TASK:\n"
            f"1. Check if the arithmetic in the proposed step is correct.\n"
            f"2. Check if the remaining numbers can still reach 24.\n"
            f"Output strictly:\n"
            f"<critique>One concise sentence explaining why this step is sound or dead-end.</critique>\n"
            f"<verdict score=\"1.0\">ACCEPT</verdict> OR <verdict score=\"0.0\">REJECT</verdict>\n"
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 100}
        }

        try:
            req = urllib.request.Request(
                self.ollama_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data.get("response", "")

            # Parse score and critique
            match_score = re.search(r'<verdict score="([\d\.]+)">', text)
            match_verdict = re.search(r'<verdict[^>]*>(.*?)</verdict>', text, re.DOTALL)
            match_critique = re.search(r'<critique>(.*?)</critique>', text, re.DOTALL)

            score = 0.5
            if match_score:
                score = float(match_score.group(1))
            elif match_verdict and "ACCEPT" in match_verdict.group(1):
                score = 1.0
            elif match_verdict and "REJECT" in match_verdict.group(1):
                score = 0.0

            critique = match_critique.group(1).strip() if match_critique else text.strip()
            is_pruned = score < self.prune_threshold

            return VerificationResult(score=score, critique=critique, is_pruned=is_pruned)
        except Exception:
            # Fallback to symbolic
            return self._evaluate_symbolic(remaining_state, candidate_step)
