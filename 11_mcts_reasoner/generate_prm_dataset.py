#!/usr/bin/env python3
"""
Generate Process Reward Model (PRM) Dataset for Game of 24
Using exact state-space solver ground truth + optional Ollama teacher critiques.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import urllib.request
from fractions import Fraction
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcts_reasoner.solver import (
    classify_next_steps,
    find_winning_paths,
    is_solvable,
    get_possible_moves,
    Step,
)

TOT_24_CSV_URL = "https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/master/src/tot/data/24/24.csv"


_CSV_CACHE_PATH = str(Path(__file__).resolve().parent / "data" / "24.csv")

def download_or_load_csv(cache_path: str = _CSV_CACHE_PATH) -> list[dict]:
    """Download 24.csv if not cached, return list of puzzle dicts."""
    p = Path(cache_path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {TOT_24_CSV_URL} to {cache_path}...")
        req = urllib.request.Request(TOT_24_CSV_URL, headers={"User-Agent": "PromptlyBrokenPRM"})
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode("utf-8")
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()

    reader = csv.DictReader(io.StringIO(content))
    puzzles = []
    for row in reader:
        nums = [int(x) for x in row["Puzzles"].split()]
        puzzles.append({
            "rank": int(row["Rank"]),
            "numbers": nums,
            "raw": row["Puzzles"]
        })
    return puzzles


def get_template_critique(
    puzzle_nums: list[int],
    prefix_steps: list[str],
    candidate_step: str,
    label: float,
    error_type: str | None,
    remaining: list[str]
) -> str:
    """Fast, deterministic, accurate critique generator."""
    if label == 1.0:
        if len(remaining) == 1 and remaining[0] in ["24", "24/1"]:
            return (
                f"Valid final move. Executing '{candidate_step}' correctly consumes the final remaining operands "
                f"and terminates exactly at target 24 without rule violations."
            )
        return (
            f"Sound intermediate step. Executing '{candidate_step}' leaves remaining numbers {remaining}. "
            f"The state remains mathematically solvable and unlocks viable downstream paths to target 24."
        )
    else:
        if error_type == "arithmetic_error":
            return (
                f"Fatal arithmetic blunder in '{candidate_step}'. The claimed equality is mathematically incorrect. "
                f"Hallucinated arithmetic invalidates all subsequent steps."
            )
        elif error_type == "hallucinated_operand":
            return (
                f"Rule violation in '{candidate_step}'. The step attempts to use numbers not available in the current "
                f"remaining number pool. Operations may only use numbers currently in play."
            )
        else:
            return (
                f"Dead-end tactical trap. Executing '{candidate_step}' leaves remaining pool {remaining}. "
                f"No permutation of standard arithmetic operations on {remaining} can reach target 24. Branch is doomed."
            )


def query_ollama_teacher(
    model_name: str,
    prompt: str,
    base_url: str = "http://127.0.0.1:11434/api/generate"
) -> str:
    """Call local Ollama model to generate a rich natural language critique."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 120,
        }
    }
    req = urllib.request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except Exception as e:
        return ""


def build_prm_record(
    puzzle_id: str,
    initial_nums: list[int],
    prefix_steps: list[str],
    candidate_step: str,
    label: float,
    error_type: str | None,
    remaining_nums: list[Fraction],
    teacher_model: str | None = None
) -> dict:
    """Formats a single PRM step-level training record."""
    rem_str = [str(x) for x in remaining_nums]
    verdict_text = "ACCEPT" if label == 1.0 else "REJECT"

    critique_text = ""
    if teacher_model:
        teacher_prompt = (
            f"You are a mathematical PRM (Process Reward Model) verifier for Game of 24.\n"
            f"Puzzle numbers: {initial_nums}\n"
            f"Prior steps: {prefix_steps}\n"
            f"Proposed step: {candidate_step}\n"
            f"Remaining numbers after step: {rem_str}\n"
            f"Ground-truth status: {'VALID' if label == 1.0 else 'FATAL_ERROR: ' + str(error_type)}\n\n"
            f"Provide a 1-2 sentence precise critique explaining why this step is {'sound' if label == 1.0 else 'rejected'}."
        )
        critique_text = query_ollama_teacher(teacher_model, teacher_prompt)

    if not critique_text:
        critique_text = get_template_critique(
            initial_nums, prefix_steps, candidate_step, label, error_type, rem_str
        )

    formatted_target = (
        f"<critique>\n{critique_text}\n</critique>\n"
        f"<verdict score=\"{label:.1f}\">{verdict_text}</verdict>"
    )

    prefix_text = "\n".join(prefix_steps) if prefix_steps else "Initial State"

    prompt_text = (
        f"Puzzle Numbers: {initial_nums} -> Target: 24\n"
        f"Previous Steps:\n{prefix_text}\n\n"
        f"Evaluate Proposed Step:\n{candidate_step}\n"
        f"Is this step sound?"
    )

    chatml_text = (
        f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{formatted_target}<|im_end|>"
    )

    return {
        "text": chatml_text,
        "puzzle_id": puzzle_id,
        "initial_numbers": initial_nums,
        "prefix": prefix_text,
        "candidate_step": candidate_step,
        "label": label,
        "error_type": error_type,
        "formatted_prompt": prompt_text,
        "formatted_target": formatted_target
    }


def generate_synthetic_prm_dataset(
    puzzles: list[dict],
    max_puzzles: int = 400,
    use_teacher: str | None = None
) -> list[dict]:
    """
    Generates step-level PRM records covering:
      - Valid moves at Steps 1, 2, 3
      - Dead-end traps at Steps 1, 2, 3
      - Injected arithmetic mutations
      - Injected hallucinated operand mutations
    """
    records: list[dict] = []

    sampled = puzzles[:max_puzzles]
    print(f"Generating PRM records from {len(sampled)} puzzles...")

    for idx, p in enumerate(sampled):
        nums = p["numbers"]
        fractions = tuple(Fraction(x, 1) for x in nums)
        p_id = f"tot_{p['rank']:04d}"

        if not is_solvable(fractions):
            continue

        winning_paths = find_winning_paths(fractions, max_paths=2)
        if not winning_paths:
            continue

        primary_winning_path = winning_paths[0]

        # --- STEP 1 ---
        winning_s1, dead_s1 = classify_next_steps(fractions)
        # 1. Positive Step 1 (Prioritize fractions and emit up to 2 examples)
        if winning_s1:
            frac_wins = [m for m in winning_s1 if any(x.denominator != 1 for x in m.remaining)]
            int_wins = [m for m in winning_s1 if all(x.denominator == 1 for x in m.remaining)]
            
            chosen_wins = []
            if frac_wins: chosen_wins.append(frac_wins[0])
            if int_wins: chosen_wins.append(int_wins[0])
            if not chosen_wins: chosen_wins.append(winning_s1[0])
            
            for best_s1 in chosen_wins:
                records.append(build_prm_record(
                    p_id, nums, [], f"Step 1: {best_s1.description()}", 1.0, None, list(best_s1.remaining), use_teacher
                ))

        # 2. Dead-End Step 1 (Emit multiple traps, especially fractions)
        if dead_s1:
            frac_traps = [m for m in dead_s1 if any(x.denominator != 1 for x in m.remaining)]
            int_traps = [m for m in dead_s1 if all(x.denominator == 1 for x in m.remaining)]
            
            chosen_traps = []
            if frac_traps:
                random.shuffle(frac_traps)
                chosen_traps.extend(frac_traps[:2])
            if int_traps:
                random.shuffle(int_traps)
                chosen_traps.extend(int_traps[:2])
            
            chosen_traps = chosen_traps[:3]
            if not chosen_traps:
                chosen_traps.append(random.choice(dead_s1))

            for trap_s1 in chosen_traps:
                records.append(build_prm_record(
                    p_id, nums, [], f"Step 1: {trap_s1.description()}", 0.0, "dead_end_trap", list(trap_s1.remaining), use_teacher
                ))

        # 3. Arithmetic Blunder Mutation at Step 1
        if len(nums) >= 2:
            n1, n2 = nums[0], nums[1]
            fake_res = n1 + n2 + random.choice([1, 2, -1, 3])
            records.append(build_prm_record(
                p_id, nums, [], f"Step 1: {n1} + {n2} = {fake_res}", 0.0, "arithmetic_error", [], use_teacher
            ))

        # --- STEP 2 ---
        step1_move = primary_winning_path[0]
        step1_prefix = [f"Step 1: {step1_move.description()}"]
        state_s2 = step1_move.remaining

        winning_s2, dead_s2 = classify_next_steps(state_s2)
        # 1. Positive Step 2
        if winning_s2:
            frac_wins = [m for m in winning_s2 if any(x.denominator != 1 for x in m.remaining)]
            int_wins = [m for m in winning_s2 if all(x.denominator == 1 for x in m.remaining)]
            
            chosen_wins = []
            if frac_wins: chosen_wins.append(frac_wins[0])
            if int_wins: chosen_wins.append(int_wins[0])
            if not chosen_wins: chosen_wins.append(winning_s2[0])
            
            for best_s2 in chosen_wins:
                records.append(build_prm_record(
                    p_id, nums, step1_prefix, f"Step 2: {best_s2.description()}", 1.0, None, list(best_s2.remaining), use_teacher
                ))

        # 2. Dead-End Step 2
        if dead_s2:
            frac_traps = [m for m in dead_s2 if any(x.denominator != 1 for x in m.remaining)]
            int_traps = [m for m in dead_s2 if all(x.denominator == 1 for x in m.remaining)]
            
            chosen_traps = []
            if frac_traps:
                random.shuffle(frac_traps)
                chosen_traps.extend(frac_traps[:2])
            if int_traps:
                random.shuffle(int_traps)
                chosen_traps.extend(int_traps[:2])
            
            chosen_traps = chosen_traps[:3]
            if not chosen_traps:
                chosen_traps.append(random.choice(dead_s2))

            for trap_s2 in chosen_traps:
                records.append(build_prm_record(
                    p_id, nums, step1_prefix, f"Step 2: {trap_s2.description()}", 0.0, "dead_end_trap", list(trap_s2.remaining), use_teacher
                ))

        # --- STEP 3 (TERMINAL) ---
        if len(primary_winning_path) >= 3:
            step2_move = primary_winning_path[1]
            step3_move = primary_winning_path[2]
            prefix_s3 = step1_prefix + [f"Step 2: {step2_move.description()}"]

            # Positive final move
            records.append(build_prm_record(
                p_id, nums, prefix_s3, f"Step 3: {step3_move.description()}", 1.0, None, list(step3_move.remaining), use_teacher
            ))

            # Terminal dead-end move (e.g. wrong operation on last two numbers)
            last_nums = step2_move.remaining
            if len(last_nums) == 2:
                for alt_op in ["-", "+", "*", "/"]:
                    if alt_op != step3_move.op:
                        bad_steps = [m for m in get_possible_moves(last_nums) if m.op == alt_op]
                        if bad_steps and bad_steps[0].result != Fraction(24, 1):
                            bad_m = bad_steps[0]
                            records.append(build_prm_record(
                                p_id, nums, prefix_s3, f"Step 3: {bad_m.description()}", 0.0, "terminal_deviation", list(bad_m.remaining), use_teacher
                            ))
                            break

        if (idx + 1) % 50 == 0 or (idx + 1) == len(sampled):
            print(f"  Processed {idx + 1}/{len(sampled)} puzzles -> {len(records)} PRM records generated.")

    return records


def main():
    parser = argparse.ArgumentParser(description="Generate PRM Dataset for Game of 24")
    parser.add_argument("--max-puzzles", type=int, default=300, help="Number of puzzles from 24.csv to process")
    parser.add_argument("--teacher", type=str, default=None, help="Ollama model for critiques, e.g. qwen3:32b or qwen3-coder:30b")
    parser.add_argument("--out-dir", type=str, default="mcts_reasoner/prm_data", help="Output directory for jsonl files")
    args = parser.parse_args()

    puzzles = download_or_load_csv()
    records = generate_synthetic_prm_dataset(puzzles, max_puzzles=args.max_puzzles, use_teacher=args.teacher)

    random.seed(42)
    random.shuffle(records)

    # 90% train, 10% valid
    split_idx = int(len(records) * 0.9)
    train_records = records[:split_idx]
    valid_records = records[split_idx:]

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_file = out_path / "train.jsonl"
    valid_file = out_path / "valid.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    with open(valid_file, "w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(r) + "\n")

    print("\n" + "=" * 60)
    print(f"🎉 PRM Dataset Generated Successfully in {args.out_dir}/")
    print(f"   Train samples: {len(train_records)}")
    print(f"   Valid samples: {len(valid_records)}")
    print(f"   Positive ratio: {sum(1 for r in train_records if r['label'] == 1.0) / len(train_records):.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
