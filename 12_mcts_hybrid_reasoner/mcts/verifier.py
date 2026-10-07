"""
Hybrid Process Reward Model (PRM) Verifier
Evaluates candidate steps and returns calibrated value score + critique.
Supports:
- Mode "symbolic": Exact DAG Oracle solver
- Mode "mlx": High-speed logit-based calibrated PRM scorer with state caching
- Mode "mlx_gen": Legacy generation-based PRM scorer
- Mode "ollama": Local Ollama API PRM scorer
"""

from __future__ import annotations
import json
import os
import re
import time
import urllib.request
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple, Sequence

from solver import is_solvable
from prm_format import (
    build_prm_chatml_prompt,
    VERDICT_PREFIX,
)
from prm_infer import score_prompt

# Module-level cache: model + tokenizer are loaded once and reused across calls.
_MLX_MODEL = None
_MLX_TOKENIZER = None
_ONE_TOKEN_ID = None
_ZERO_TOKEN_ID = None


class VerificationResult(NamedTuple):
    score: float            # Scaled 0.0 to 1.0 (calibrated probability in MLX mode)
    critique: str           # Verbal explanation
    is_pruned: bool         # Whether MCTS should prune this branch


class PRMVerifier:
    """
    Evaluates individual reasoning steps using:
    - Mode A: Exact Symbolic Oracle (ideal for testing MCTS dynamics and speed)
    - Mode B: MLX Logit PRM (one forward pass, calibrated p(ACCEPT), state cache)
    - Mode C: Local Ollama / MLX Generation PRM
    """

    # Default adapter path relative to repo root (no redundant nested mcts_reasoner directory)
    _DEFAULT_ADAPTER_PATH: str = str(
        Path(__file__).resolve().parent.parent / "prm_adapters_v2"
    )

    def __init__(
        self,
        mode: str = "symbolic",  # "symbolic" | "mlx" | "mlx_gen" | "ollama"
        prune_threshold: float = 0.25,
        model_name: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
        ollama_url: str = "http://127.0.0.1:11434/api/generate",
        adapter_path: str | None = None,
        terminal_shortcut: bool = True,
        use_scratch: bool | None = None,
    ):
        self.mode = mode
        self.prune_threshold = prune_threshold
        self.model_name = model_name
        self.ollama_url = ollama_url
        # Resolution order: explicit arg > PRM_ADAPTER_PATH env (set by benchmark --adapter) > v2 default
        self.adapter_path = adapter_path or os.environ.get("PRM_ADAPTER_PATH") or self._DEFAULT_ADAPTER_PATH
        self.terminal_shortcut = terminal_shortcut
        # v3 scratchpad protocol: model writes <scratch>..</scratch> for 2/3-number states before the verdict.
        # Must be True only with a v3-trained adapter; default off keeps v2 behaviour.
        if use_scratch is None:
            use_scratch = os.environ.get("PRM_USE_SCRATCH", "0").lower() in ("1", "true", "yes")
        self.use_scratch = use_scratch

        # Telemetry
        self.calls: int = 0
        self.cache_hits: int = 0
        self.model_ms: float = 0.0
        self.scratch_tokens: int = 0
        self.scratch_failures: int = 0

        # Cache keyed by (history_tuple, candidate_step)
        self._cache: dict[tuple[tuple[str, ...], str], VerificationResult] = {}

    def evaluate_step(
        self,
        initial_numbers: list[int],
        history: list[str],
        candidate_step: str,
        remaining_state: tuple[Fraction, ...]
    ) -> VerificationResult:
        """Main entry point to evaluate a candidate step."""
        self.calls += 1

        # Check terminal shortcut first (User Decision #2)
        if self.terminal_shortcut and len(remaining_state) == 1:
            is_sound = (remaining_state[0] == Fraction(24, 1))
            score = 1.0 if is_sound else 0.0
            critique = (
                f"Terminal state verified: {remaining_state[0]} == 24"
                if is_sound
                else f"Terminal state failed: {remaining_state[0]} != 24"
            )
            return VerificationResult(score=score, critique=critique, is_pruned=(score < self.prune_threshold))

        cache_key = (tuple(history), candidate_step.strip())
        if cache_key in self._cache:
            self.cache_hits += 1
            return self._cache[cache_key]

        if self.mode == "symbolic":
            res = self._evaluate_symbolic(remaining_state, candidate_step)
        elif self.mode == "hybrid":
            # Hybrid architecture: exact symbolic check for terminal / pre-terminal (<= 2 numbers),
            # neural MLX PRM for intermediate multi-branch ranking (>= 3 numbers).
            if len(remaining_state) <= 2:
                res = self._evaluate_symbolic(remaining_state, candidate_step)
            else:
                res = self._evaluate_mlx_logit(initial_numbers, history, candidate_step, remaining_state)
        elif self.mode == "mlx":
            res = self._evaluate_mlx_logit(initial_numbers, history, candidate_step, remaining_state)
        elif self.mode == "mlx_gen":
            res = self._evaluate_mlx_gen(initial_numbers, history, candidate_step, remaining_state)
        elif self.mode == "ollama":
            res = self._evaluate_ollama(initial_numbers, history, candidate_step, remaining_state)
        else:
            res = self._evaluate_symbolic(remaining_state, candidate_step)

        self._cache[cache_key] = res
        return res

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

    def _ensure_mlx_loaded(self) -> bool:
        """Loads MLX model, tokenizer, and resolves logit score token IDs."""
        global _MLX_MODEL, _MLX_TOKENIZER, _ONE_TOKEN_ID, _ZERO_TOKEN_ID

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
                print(f"[PRM] MLX load failed ({exc}).")
                return False

        if _ONE_TOKEN_ID is None or _ZERO_TOKEN_ID is None:
            token_1 = _MLX_TOKENIZER.encode("1")
            token_0 = _MLX_TOKENIZER.encode("0")
            _ONE_TOKEN_ID = token_1[0] if len(token_1) == 1 else _MLX_TOKENIZER.convert_tokens_to_ids("1")
            _ZERO_TOKEN_ID = token_0[0] if len(token_0) == 1 else _MLX_TOKENIZER.convert_tokens_to_ids("0")

        return True

    def _evaluate_mlx_logit(
        self,
        initial_numbers: list[int],
        history: list[str],
        candidate_step: str,
        remaining_state: tuple[Fraction, ...]
    ) -> VerificationResult:
        """Evaluates step using verdict-logit scoring for p(ACCEPT).

        With use_scratch=True (v3 adapters) a scratchpad is generated first for 2/3-number states.
        """
        if not self._ensure_mlx_loaded():
            return self._evaluate_symbolic(remaining_state, candidate_step)

        chatml_prompt = build_prm_chatml_prompt(initial_numbers, history, candidate_step)

        t0 = time.perf_counter()
        try:
            out = score_prompt(
                _MLX_MODEL,
                _MLX_TOKENIZER,
                chatml_prompt,
                n_remaining=len(remaining_state),
                one_token_id=_ONE_TOKEN_ID,
                zero_token_id=_ZERO_TOKEN_ID,
                use_scratch=self.use_scratch,
            )
        except Exception as exc:
            print(f"[PRM] Logit calculation error ({exc}). Falling back to symbolic.")
            return self._evaluate_symbolic(remaining_state, candidate_step)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.model_ms += elapsed_ms
        self.scratch_tokens += out.scratch_tokens
        if not out.scratch_ok:
            self.scratch_failures += 1

        prob_accept = out.prob_accept
        critique = f"PRM p(solvable)={prob_accept:.3f}"
        if out.scratch:
            critique += "\n" + out.scratch
        is_pruned = prob_accept < self.prune_threshold

        return VerificationResult(score=prob_accept, critique=critique, is_pruned=is_pruned)

    def explain_step(
        self,
        initial_numbers: list[int],
        history: list[str],
        candidate_step: str,
    ) -> str:
        """Generates natural language critique on-demand for UI/solution presentation."""
        if not self._ensure_mlx_loaded():
            return "Step verified by symbolic solver."

        from mlx_lm import generate
        full_prompt = build_prm_chatml_prompt(initial_numbers, history, candidate_step)
        try:
            out = generate(
                _MLX_MODEL, _MLX_TOKENIZER, prompt=full_prompt,
                max_tokens=(400 if self.use_scratch else 80), verbose=False,
            )
            match = re.search(r"<critique>(.*?)</critique>", out, re.DOTALL)
            if match:
                return match.group(1).strip()
            # Truncate at </verdict> or </critique>
            cleaned = out.split("<|im_end|>")[0].strip()
            return cleaned
        except Exception as e:
            return f"Explanation unavailable: {e}"

    def _evaluate_mlx_gen(
        self,
        initial_numbers: list[int],
        history: list[str],
        candidate_step: str,
        remaining_state: tuple[Fraction, ...]
    ) -> VerificationResult:
        """Legacy generation-based verifier with early stopping."""
        if not self._ensure_mlx_loaded():
            return self._evaluate_symbolic(remaining_state, candidate_step)

        full_prompt = build_prm_chatml_prompt(initial_numbers, history, candidate_step)

        try:
            from mlx_lm import generate
            response = generate(
                _MLX_MODEL,
                _MLX_TOKENIZER,
                prompt=full_prompt,
                max_tokens=40,
                verbose=False,
            )
        except Exception as exc:
            return self._evaluate_symbolic(remaining_state, candidate_step)

        score_match = re.search(r'<verdict score="([\d.]+)">', response)
        verdict_match = re.search(r'<verdict[^>]*>(.*?)</verdict>', response, re.DOTALL)
        critique_match = re.search(r'<critique>(.*?)</critique>', response, re.DOTALL)

        if score_match:
            score = float(score_match.group(1))
        elif verdict_match and "ACCEPT" in verdict_match.group(1).upper():
            score = 1.0
        elif verdict_match and "REJECT" in verdict_match.group(1).upper():
            score = 0.0
        else:
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
        """Evaluates step using local LLM prompt via Ollama API."""
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
            f"<verdict score=\"1.0\">ACCEPT</verdict> OR <verdict score=\"0.0\">REJECT</verdict>\n"
            f"<critique>One concise sentence explaining why this step is sound or dead-end.</critique>\n"
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 80}
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
            return self._evaluate_symbolic(remaining_state, candidate_step)
