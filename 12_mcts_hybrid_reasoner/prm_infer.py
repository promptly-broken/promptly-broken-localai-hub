"""
prm_infer.py — Shared v3 inference: (optional) scratchpad generation, then verdict-logit scoring.

Used by BOTH eval_prm.py and mcts/verifier.py so offline gates measure exactly what MCTS runs.

Protocol (must match generate_prm_dataset_v3.py):
  * resulting state has 2 or 3 numbers -> model first writes <scratch>...</scratch>,
    then we prefill  "\n" + '<verdict score="'  and read p('1') vs p('0').
  * otherwise (1 or 4+ numbers)       -> prefill '<verdict score="' directly (v2 behaviour).
"""

from __future__ import annotations

from typing import NamedTuple

from prm_format import VERDICT_PREFIX
from prm_scratch import SCRATCH_CLOSE, SCRATCH_OPEN, needs_scratchpad

# Longest legal scratchpad is ~160 tokens; cap well above to catch runaways.
MAX_SCRATCH_TOKENS = 260


class ScoreOutput(NamedTuple):
    prob_accept: float
    scratch: str          # generated scratchpad text ('' if none was needed)
    scratch_tokens: int   # number of generated tokens
    scratch_ok: bool      # False if a scratchpad was required but never closed


def generate_scratchpad(model, tokenizer, prompt: str, max_tokens: int = MAX_SCRATCH_TOKENS) -> tuple[str, int, bool]:
    """Greedy-generate until </scratch>. Returns (text_including_close_tag, n_tokens, closed)."""
    from mlx_lm import stream_generate

    # Force the scratchpad to start: the model was trained to open with <scratch>\n right after the assistant tag.
    text = SCRATCH_OPEN + "\n"
    n_tok = 0
    closed = False
    for resp in stream_generate(model, tokenizer, prompt=prompt + SCRATCH_OPEN + "\n", max_tokens=max_tokens):
        text += resp.text
        n_tok = resp.generation_tokens
        if SCRATCH_CLOSE in text:
            text = text[: text.index(SCRATCH_CLOSE) + len(SCRATCH_CLOSE)]
            closed = True
            break
    return text, n_tok, closed


def score_prompt(
    model,
    tokenizer,
    chatml_prompt: str,
    n_remaining: int,
    one_token_id: int,
    zero_token_id: int,
    use_scratch: bool = True,
) -> ScoreOutput:
    """Probability of ACCEPT for a ChatML prompt that ends right after '<|im_start|>assistant\\n'."""
    import mlx.core as mx

    scratch, n_tok, ok = "", 0, True
    prompt = chatml_prompt
    if use_scratch and needs_scratchpad(n_remaining):
        scratch, n_tok, ok = generate_scratchpad(model, tokenizer, chatml_prompt)
        prompt = chatml_prompt + scratch + "\n"

    full = prompt + VERDICT_PREFIX
    tokens = mx.array(tokenizer.encode(full))[None]
    logits = model(tokens)[0, -1, :]
    pair = mx.array([one_token_id, zero_token_id])
    pair_logits = logits[pair].astype(mx.float32)
    prob = float(mx.softmax(pair_logits)[0].item())
    return ScoreOutput(prob, scratch, n_tok, ok)
