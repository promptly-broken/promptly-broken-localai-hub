"""
prm_format.py — Unified ChatML prompt formatting for PRM training and inference.
Serves as the single source of truth across dataset generation, verifier, and eval harness.
"""

from __future__ import annotations
import re
from typing import Sequence

VERDICT_PREFIX = '<verdict score="'


def format_step_prefix(history: Sequence[str]) -> str:
    """Formats history into the 'Previous Steps:' text block."""
    return "\n".join(history) if history else "Initial State"


def ensure_step_numbered(candidate_step: str, step_num: int) -> str:
    """Ensures candidate step has 'Step N:' prefix without double-prefixing."""
    stripped = candidate_step.strip()
    if not re.match(r"^Step\s+\d+:", stripped):
        return f"Step {step_num}: {stripped}"
    return stripped


def build_prm_prompt_body(
    initial_numbers: Sequence[int],
    history: Sequence[str],
    candidate_step: str,
) -> str:
    """Builds the raw prompt text body without ChatML tags."""
    step_num = len(history) + 1
    prefix_text = format_step_prefix(history)
    candidate_prefixed = ensure_step_numbered(candidate_step, step_num)

    return (
        f"Puzzle Numbers: {list(initial_numbers)} -> Target: 24\n"
        f"Previous Steps:\n{prefix_text}\n\n"
        f"Evaluate Proposed Step:\n{candidate_prefixed}\n"
        f"Is this step sound?"
    )


def build_prm_chatml_prompt(
    initial_numbers: Sequence[int],
    history: Sequence[str],
    candidate_step: str,
) -> str:
    """Wraps prompt body in ChatML user/assistant turn delimiters."""
    body = build_prm_prompt_body(initial_numbers, history, candidate_step)
    return f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n"


def build_prm_completion(
    label: float,
    critique: str,
) -> str:
    """Formats the assistant completion with <verdict> followed by <critique> and <|im_end|>."""
    score_str = f"{label:.1f}"
    verdict_text = "ACCEPT" if label == 1.0 else "REJECT"
    cleaned_critique = critique.strip()

    return (
        f'<verdict score="{score_str}">{verdict_text}</verdict>\n'
        f'<critique>\n{cleaned_critique}\n</critique><|im_end|>'
    )
