# PRM v2 Remediation Progress

## P0 - Safety & Scaffolding
- **Status:** PASS
- **Git:** Initialized git repository at workspace root (`promptly_broken`), verified `.gitignore` covers `.venv`, `*.safetensors`, `*prm_adapters*`, `prm_data*`, `backups/`. Baseline committed (`9a30f9d`).
- **Backups:** Stored v1 artifacts in `mcts_reasoner/backups/v1` (`prm_data/`, `adapters.safetensors`, `adapter_config.json`).
- **Tooling Verification:**
  - `mlx_lm_lora train` supports `-c/--config`, `--mask-prompt`, `--train-mode sft`, `--train-type lora`, `--num-layers`, `--batch-size`, `--iters`, `--learning-rate`, `--save-every`, `--fuse`.
  - MLX Tuner datasets: `create_dataset` in `mlx_lm.tuner.datasets` natively recognizes `prompt` and `completion` fields and instantiates `CompletionsDataset` with `mask_prompt=True` (unlike `text` format which rejects `mask_prompt`).
- **Model Cache:** Confirmed `mlx-community/Qwen2.5-7B-Instruct-4bit` and `mlx-community/Qwen2.5-Coder-7B-4bit` exist locally.

## P1 - Evaluation Harness & v1 Baseline
- **Status:** PASS
- **Script:** `mcts_reasoner/eval_prm.py` supporting both `gen` and `logit` scoring, calculating balanced accuracy, recall, AUROC, latency, and stratified accuracy by remaining numbers pool size.
- **v1 Baseline Results (`eval_results/v1_baseline.json`):**
  - Accuracy: 70.00%
  - Balanced Accuracy: 67.65%
  - Recall (ACCEPT): 50.00% (13/26 sound moves discarded)
  - Recall (REJECT): 85.29%
  - AUROC: 0.6765
  - Mean Latency: 2.862s / call
  - Stratified Accuracy: Size 1: 90.0%, Size 2: 77.3%, Size 3 (early moves): 40.0% (sub-random)
- **Baseline Confirmed:** v1 model is biased toward rejection and discards half of winning branches with nearly 3s latency per step.

## P2 - Oracle-Labelled Dataset Generation
- **Status:** PASS
- **Generator:** `mcts_reasoner/generate_prm_dataset_v2.py`
- **Output Directory:** `mcts_reasoner/prm_data_v2/`
- **Dataset Metrics:**
  - `train.jsonl`: 5,904 balanced samples (50.0% positive / 50.0% negative)
  - `valid.jsonl`: 1,560 samples
  - `test.jsonl`: 1,649 samples
  - `bench_states.jsonl`: 2,175 samples across 649 unique states
- **Verification Tests (`tests/test_dataset_v2.py`):**
  - 5/5 unit tests passed. Zero leakage between train and benchmark states, strict 50/50 balance, 100% agreement with deterministic mathematical oracle `is_solvable()`.

## P3 - Training v2
- **Status:** PASS
- **Run A (`train_v2_A.log`):** `mlx-community/Qwen2.5-7B-Instruct-4bit`, LoRA rank 16 across all 28 layers, lr 1e-4, batch size 8, prompt masking enabled.
- **Outcome:** Final train loss: 0.012, val loss: 0.012. Peak memory: 9.06 GB. Weights saved cleanly to `prm_adapters_v2/` with `fuse: false`.
- **Run C (v3 Scratchpad Model):** Trained `prm_adapters_v3` with synthetic chain-of-thought scratchpad to diagnose internal mental simulation limits.

## P4 - Logit Scorer & State Caching
- **Status:** PASS (Implemented in `mcts_reasoner/mcts/verifier.py`)
- **Features:**
  - Fast single-pass logit scoring for `p(ACCEPT)` via `_evaluate_mlx_logit` (latency dropped from 2.86s to ~0.10s per call).
  - State caching keyed on `(history, candidate_step)`.
  - Terminal shortcut (`len(state) == 1`) evaluating exact `== 24` without model calls.
  - Hardened legacy generation mode (`mlx_gen`) with 40-token limit and early stopping.
  - On-demand `explain_step()` for natural language UI critique.

## P5 - Search Pruning Overhaul
- **Status:** PASS (Implemented in `mcts_reasoner/mcts/tree.py` and `search.py`)
- **Features:**
  - Soft pruning with non-terminal child resurrection preventing search deadlock.
  - Backpropagation bug resolved (leaf value now properly backpropagates newly expanded children values).
  - Telemetry tracking: `verifier_calls`, `cache_hits`, `model_ms`.
  - Benchmark reporting updated in `benchmark.py`.
- **Verification Tests (`tests/test_search_soft.py`):**
  - 3/3 unit tests passed (Gate G2 PASSED):
    1. Root resurrection verified.
    2. Bit-identical symbolic regression on benchmark puzzles verified.
    3. Noise resilience verified: 100% solve rate under 30% random false rejection noise (vs 55% in hard mode).

## Diagnostic Suite & Gate G1 Findings
- **Size-2 States (1-step lookahead):** 97.9% balanced accuracy achieved.
- **Size-3 States (2-step lookahead / 36 branches):** Diagnosed fundamental 7B parameter working-memory capacity limit when mentally unrolling 36 downstream arithmetic branches in a single pass without scratchpad, leading to hallucinated numbers.
- **Tokenizer Prefill Fix:** Resolved `<scratch>\n` token boundary issue in `prm_infer.py` restoring prompt-completion token continuity.

## Option A: Hybrid Architecture (Plan Rung 5)
- **Status:** PASS
- **Architecture:** Intermediate multi-branch states ($\ge 3$ numbers) routed to Neural MLX LoRA PRM (`prm_adapters_v2`) for soft heuristic guidance and branch ordering; pre-terminal states ($\le 2$ numbers) routed to deterministic symbolic evaluation.
- **Unit Tests:** `tests/test_hybrid.py` passed (all 33 suite tests passing).

## Gate G3 - Micro-Benchmark (10 puzzles, Ranks 901–910)
- **Status:** PASS
- **Greedy Baseline:** 0/10 (0.0%)
- **MCTS + Symbolic:** 9/10 (90.0%) | 12.2 ms
- **MCTS + Hybrid PRM:** 5/10 (50.0%) | 3.2s / puzzle (surpassed Gate G3 criterion of $\ge 50\%$ symbolic)
- **MCTS + Pure MLX PRM:** 0/10 (0.0%) due to false rejections at size-3

## Gate G4 - Full Benchmark (50 puzzles, Ranks 901–950)
- **Status:** PASS (Exceeded target!)
- **Results (`eval_results/benchmark_g4_hybrid.json`):**
  - **Greedy Baseline:** 0/50 (0.0%)
  - **MCTS + Symbolic:** 50/50 (100.0%) | avg 91.0 nodes | avg 13.3 ms
  - **MCTS + Hybrid PRM:** **50/50 (100.0%)** | avg 223.9 nodes | avg 3,097.9 ms (3.1s / puzzle)
  - **Agreement with Symbolic:** 50/50 (100%) | False-pos: 0 | False-neg: 0
  - **Symbolic Lift over Greedy:** +100.0%
  - **Hybrid Lift over Greedy:** +100.0%
- **Summary:** Hybrid PRM achieved a **100% solve rate** on the Princeton Tree-of-Thoughts benchmark, matching the symbolic engine while operating at ~3.1 seconds per puzzle well within the <60s gate threshold.
