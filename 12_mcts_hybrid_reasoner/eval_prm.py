#!/usr/bin/env python3
"""
eval_prm.py — Standardized Evaluation Harness for Process Reward Models (PRM)
Measures Accuracy, Balanced Accuracy, Precision/Recall, AUROC, latency, and stratified performance.
Supports both generation-based (v1) and logit-based (v2) scoring.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Set offline mode by default
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def compute_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Computes AUROC using rank-sum / Mann-Whitney statistic without sklearn."""
    pos = y_scores[y_true == 1.0]
    neg = y_scores[y_true == 0.0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Count pairs (pos > neg) + 0.5 * (pos == neg)
    all_scores = np.concatenate([pos, neg])
    all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(all_scores) + 1)

    # Handle ties by assigning average rank
    unique_scores, counts = np.unique(all_scores, return_counts=True)
    for score, count in zip(unique_scores, counts):
        if count > 1:
            tie_mask = all_scores == score
            ranks[tie_mask] = ranks[tie_mask].mean()

    pos_rank_sum = ranks[:n_pos].sum()
    u = pos_rank_sum - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def extract_remaining_count(row: dict[str, Any]) -> int | None:
    """Infers number of remaining numbers from candidate_step or remaining state."""
    if "n_numbers" in row:
        return int(row["n_numbers"])
    if "remaining_state" in row:
        return len(row["remaining_state"])
    if "candidate_step" in row:
        m = re.search(r"Remaining:\s*\[(.*?)\]", str(row["candidate_step"]))
        if m:
            parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
            return len(parts)
    return None


def run_eval(
    data_path: Path,
    adapter_path: str | None,
    model_name: str,
    n_samples: int | None,
    scorer: str,
    output_path: Path | None,
    seed: int = 0,
) -> dict[str, Any]:
    print(f"\n[EVAL] Loading evaluation data from: {data_path}")
    if data_path.is_dir():
        candidate_file = data_path / "test.jsonl"
        if not candidate_file.exists():
            candidate_file = data_path / "valid.jsonl"
        data_file = candidate_file
    else:
        data_file = data_path

    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_file}")

    with open(data_file, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    print(f"[EVAL] Total available rows: {len(rows)}")
    random.seed(seed)
    random.shuffle(rows)
    if n_samples and n_samples < len(rows):
        rows = rows[:n_samples]
    print(f"[EVAL] Selected evaluation samples: {len(rows)}")

    # Load MLX model and tokenizer
    from mlx_lm import load

    print(f"[EVAL] Loading model '{model_name}' with adapter '{adapter_path}'...")
    model, tokenizer = load(model_name, adapter_path=adapter_path)
    vocab_sz = getattr(tokenizer, "vocab_size", getattr(tokenizer._tokenizer, "vocab_size", "unknown"))
    print(f"[EVAL] Model loaded successfully. Tokenizer vocab size: {vocab_sz}")

    latencies: list[float] = []
    y_true: list[float] = []
    y_pred_bin: list[float] = []
    y_pred_score: list[float] = []
    remaining_counts: list[int | None] = []
    scratch_tokens: list[int] = []
    scratch_failures = 0

    # Identify verdict token IDs if a logit-based scorer is requested
    yes_token_id = None
    no_token_id = None
    if scorer in ("logit", "scratch"):
        # Look for tokens '1' and '0' after <verdict score="
        token_1 = tokenizer.encode("1")
        token_0 = tokenizer.encode("0")
        print(f"[EVAL] Logit scorer token '1' IDs: {token_1}, '0' IDs: {token_0}")
        if len(token_1) == 1 and len(token_0) == 1:
            yes_token_id = token_1[0]
            no_token_id = token_0[0]
        else:
            # Fallback to single token lookup
            yes_token_id = tokenizer.convert_tokens_to_ids("1")
            no_token_id = tokenizer.convert_tokens_to_ids("0")
        print(f"[EVAL] Using token IDs: score='1': {yes_token_id}, score='0': {no_token_id}")

    print(f"[EVAL] Starting evaluation in mode '{scorer}'...")
    from mlx_lm import generate
    import mlx.core as mx

    for idx, row in enumerate(rows):
        # Determine ground truth label
        true_label = float(row.get("label", 1.0 if row.get("completion", "").startswith("Yes") else 0.0))
        y_true.append(true_label)
        n_rem = extract_remaining_count(row)
        remaining_counts.append(n_rem)

        # Prepare prompt
        if "formatted_prompt" in row:
            prompt_text = row["formatted_prompt"]
            full_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        elif "prompt" in row:
            full_prompt = row["prompt"]
        elif "text" in row:
            # Split before assistant completion
            text = row["text"]
            if "<|im_start|>assistant\n" in text:
                full_prompt = text.split("<|im_start|>assistant\n")[0] + "<|im_start|>assistant\n"
            else:
                full_prompt = text
        else:
            raise ValueError(f"Unknown row format: {list(row.keys())}")

        t0 = time.perf_counter()

        if scorer == "gen":
            out = generate(model, tokenizer, prompt=full_prompt, max_tokens=150, verbose=False)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)

            score_match = re.search(r'<verdict score="([\d.]+)">', out)
            verdict_match = re.search(r'<verdict[^>]*>(.*?)</verdict>', out, re.DOTALL)
            if score_match:
                score = float(score_match.group(1))
            elif verdict_match and "ACCEPT" in verdict_match.group(1).upper():
                score = 1.0
            elif verdict_match and "REJECT" in verdict_match.group(1).upper():
                score = 0.0
            else:
                score = 0.0  # Default unparseable to reject

            y_pred_score.append(score)
            y_pred_bin.append(1.0 if score >= 0.5 else 0.0)

        elif scorer == "scratch":
            from prm_infer import score_prompt

            out = score_prompt(
                model, tokenizer, full_prompt,
                n_remaining=n_rem if n_rem is not None else 0,
                one_token_id=yes_token_id, zero_token_id=no_token_id,
            )
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            scratch_tokens.append(out.scratch_tokens)
            if not out.scratch_ok:
                scratch_failures += 1
            y_pred_score.append(out.prob_accept)
            y_pred_bin.append(1.0 if out.prob_accept >= 0.5 else 0.0)

        elif scorer == "logit":
            # Prefill prompt + <verdict score="
            verdict_prefix = '<verdict score="'
            if not full_prompt.endswith(verdict_prefix):
                full_prompt = full_prompt + verdict_prefix
            
            prompt_tokens = mx.array(tokenizer.encode(full_prompt))[None]
            logits = model(prompt_tokens)[0, -1, :]
            pair = mx.array([yes_token_id, no_token_id])
            pair_logits = logits[pair].astype(mx.float32)
            prob_accept = float(mx.softmax(pair_logits)[0].item())
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)

            y_pred_score.append(prob_accept)
            y_pred_bin.append(1.0 if prob_accept >= 0.5 else 0.0)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(rows):
            print(f"  Processed {idx + 1}/{len(rows)} samples (avg latency: {np.mean(latencies):.2f}s)...")

    # Compute aggregate metrics
    y_true_arr = np.array(y_true)
    y_pred_bin_arr = np.array(y_pred_bin)
    y_pred_score_arr = np.array(y_pred_score)

    tp = int(np.sum((y_true_arr == 1.0) & (y_pred_bin_arr == 1.0)))
    tn = int(np.sum((y_true_arr == 0.0) & (y_pred_bin_arr == 0.0)))
    fp = int(np.sum((y_true_arr == 0.0) & (y_pred_bin_arr == 1.0)))
    fn = int(np.sum((y_true_arr == 1.0) & (y_pred_bin_arr == 0.0)))

    n_pos = tp + fn
    n_neg = tn + fp
    total = len(y_true_arr)

    accuracy = (tp + tn) / total if total > 0 else 0.0
    recall_accept = tp / n_pos if n_pos > 0 else 0.0
    recall_reject = tn / n_neg if n_neg > 0 else 0.0
    balanced_acc = (recall_accept + recall_reject) / 2.0
    precision_accept = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_accept = (
        2 * precision_accept * recall_accept / (precision_accept + recall_accept)
        if (precision_accept + recall_accept) > 0
        else 0.0
    )

    auroc = compute_auroc(y_true_arr, y_pred_score_arr)

    # Accuracy stratified by number of remaining numbers
    stratified_acc = {}
    for count_val in [1, 2, 3, 4]:
        mask = np.array([r == count_val for r in remaining_counts])
        if np.any(mask):
            sub_true = y_true_arr[mask]
            sub_pred = y_pred_bin_arr[mask]
            stratified_acc[f"size_{count_val}"] = {
                "count": int(np.sum(mask)),
                "accuracy": float(np.mean(sub_true == sub_pred)),
                "mean_latency_sec": round(float(np.mean(np.array(latencies)[mask])), 3),
            }

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "adapter": adapter_path,
        "scorer": scorer,
        "dataset": str(data_file),
        "total_samples": total,
        "class_distribution": {"pos_accept": n_pos, "neg_reject": n_neg},
        "accuracy": round(float(accuracy), 4),
        "balanced_accuracy": round(float(balanced_acc), 4),
        "recall_accept": round(float(recall_accept), 4),
        "recall_reject": round(float(recall_reject), 4),
        "precision_accept": round(float(precision_accept), 4),
        "f1_accept": round(float(f1_accept), 4),
        "auroc": round(float(auroc), 4),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "stratified_by_remaining_size": stratified_acc,
        "latency": {
            "mean_sec": round(float(np.mean(latencies)), 3),
            "median_sec": round(float(np.median(latencies)), 3),
            "p95_sec": round(float(np.percentile(latencies, 95)), 3),
        },
    }
    if scorer == "scratch":
        nz = [t for t in scratch_tokens if t > 0]
        results["scratch"] = {
            "rows_with_scratchpad": len(nz),
            "mean_tokens": round(float(np.mean(nz)), 1) if nz else 0.0,
            "max_tokens": int(max(nz)) if nz else 0,
            "unclosed_failures": scratch_failures,
        }

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS".center(60))
    print("=" * 60)
    print(f"Total Samples:       {total} (ACCEPT={n_pos}, REJECT={n_neg})")
    print(f"Accuracy:            {accuracy * 100:.2f}%")
    print(f"Balanced Accuracy:   {balanced_acc * 100:.2f}%")
    print(f"Recall (ACCEPT):     {recall_accept * 100:.2f}% ({tp}/{n_pos})")
    print(f"Recall (REJECT):     {recall_reject * 100:.2f}% ({tn}/{n_neg})")
    print(f"Precision (ACCEPT):  {precision_accept * 100:.2f}%")
    print(f"F1 (ACCEPT):         {f1_accept:.4f}")
    print(f"AUROC:               {auroc:.4f}")
    print(f"Confusion Matrix:    TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Latency per call:    mean={results['latency']['mean_sec']}s, p95={results['latency']['p95_sec']}s")
    if stratified_acc:
        print(f"Accuracy by size:    {stratified_acc}")
    if "scratch" in results:
        print(f"Scratchpad stats:    {results['scratch']}")
    print("=" * 60 + "\n")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[EVAL] Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PRM on validation/test set")
    parser.add_argument("--data", type=str, required=True, help="Path to jsonl file or directory")
    parser.add_argument(
        "--adapter",
        type=str,
        default="mcts_reasoner/prm_adapters",
        help="Path to LoRA adapters directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2.5-Coder-7B-4bit",
        help="HuggingFace model name or local path",
    )
    parser.add_argument("-n", "--n-samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument(
        "--scorer", choices=["gen", "logit", "scratch"], default="gen",
        help="gen=v1 generation, logit=one-pass verdict logit (v2), scratch=scratchpad then verdict logit (v3)",
    )
    parser.add_argument("--out", type=str, default=None, help="Output path for results JSON")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out) if args.out else None

    run_eval(
        data_path=data_path,
        adapter_path=args.adapter,
        model_name=args.model,
        n_samples=args.n_samples,
        scorer=args.scorer,
        output_path=out_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
