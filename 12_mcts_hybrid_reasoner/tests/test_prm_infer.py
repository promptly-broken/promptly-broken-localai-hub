"""
test_prm_infer.py — Tests for scratchpad-then-logit scoring using a fake model/tokenizer
(no weights loaded; runs in milliseconds). unittest only.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx

from prm_format import VERDICT_PREFIX
from prm_infer import score_prompt

PROMPT = "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n"


class FakeTokenizer:
    def __init__(self):
        self.encoded: list[str] = []

    def encode(self, text):
        self.encoded.append(text)
        return [1, 2, 3]


class FakeModel:
    """Returns logits that strongly favour token id 16 ('1') over 15 ('0') at the last position."""

    def __call__(self, tokens):
        logits = mx.zeros((1, tokens.shape[1], 32))
        logits[0, -1, 16] = 5.0
        return logits


def fake_stream(chunks):
    def _gen(model, tokenizer, prompt, max_tokens=256, **kw):
        for i, c in enumerate(chunks, 1):
            yield SimpleNamespace(text=c, generation_tokens=i)
    return _gen


class TestScoreProm(unittest.TestCase):
    def run_score(self, n, chunks=None, use_scratch=True):
        tok = FakeTokenizer()
        with mock.patch("mlx_lm.stream_generate", side_effect=fake_stream(chunks or [])) as sg:
            out = score_prompt(FakeModel(), tok, PROMPT, n, one_token_id=16, zero_token_id=15, use_scratch=use_scratch)
        return out, tok, sg

    def test_no_scratch_for_size_1_and_4(self):
        for n in (1, 4):
            out, tok, sg = self.run_score(n, ["x"])
            sg.assert_not_called()
            self.assertEqual(out.scratch, "")
            self.assertEqual(tok.encoded[-1], PROMPT + VERDICT_PREFIX)
            self.assertGreater(out.prob_accept, 0.99)

    def test_scratch_generated_for_size_3(self):
        out, tok, sg = self.run_score(3, ["[1, 2, 3]\nreach", "able: yes\n</scr", "atch>garbage after"])
        sg.assert_called_once()
        self.assertTrue(out.scratch_ok)
        self.assertTrue(out.scratch.startswith("<scratch>\n"))
        self.assertTrue(out.scratch.endswith("</scratch>"))
        self.assertNotIn("garbage", out.scratch)
        self.assertEqual(tok.encoded[-1], PROMPT + out.scratch + "\n" + VERDICT_PREFIX)

    def test_scratch_for_size_2(self):
        out, tok, sg = self.run_score(2, ["[3, 8]: 11\nhit 3*8=24\n</scratch>"])
        sg.assert_called_once()
        self.assertTrue(out.scratch_ok)

    def test_unclosed_scratch_flagged(self):
        out, _, _ = self.run_score(3, ["\nloop", "loop", "loop"])
        self.assertFalse(out.scratch_ok)

    def test_disabled_scratch_skips_generation(self):
        out, tok, sg = self.run_score(3, ["x"], use_scratch=False)
        sg.assert_not_called()
        self.assertEqual(tok.encoded[-1], PROMPT + VERDICT_PREFIX)


class TestVerifierConfig(unittest.TestCase):
    def test_env_flags(self):
        from mcts.verifier import PRMVerifier

        with mock.patch.dict("os.environ", {"PRM_USE_SCRATCH": "1", "PRM_ADAPTER_PATH": "/tmp/x"}):
            v = PRMVerifier(mode="symbolic")
            self.assertTrue(v.use_scratch)
            self.assertEqual(v.adapter_path, "/tmp/x")
        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("PRM_USE_SCRATCH", None)
            os.environ.pop("PRM_ADAPTER_PATH", None)
            v = PRMVerifier(mode="symbolic")
            self.assertFalse(v.use_scratch)
            self.assertTrue(v.adapter_path.endswith("prm_adapters_v2"))


if __name__ == "__main__":
    unittest.main()
