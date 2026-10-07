#!/usr/bin/env python3
"""
generate_prm_dataset_v2.py — PRM v2 Oracle-Labelled Dataset Generator
Generates high-volume, exact oracle-labelled PRM examples for Game of 24.
Key features:
- Uses exact solver oracle is_solvable() and find_winning_paths() (no 32B teacher)
- Strictly excludes benchmark puzzles (ranks 901-950) from training
- Generates disjoint state-based splits (train/valid/test)
- Outputs bench_states.jsonl for evaluation on unseen benchmark test states
- Uses prompt/completion JSON format compatible with mlx_lm.tuner mask_prompt
- Formats completions with <verdict> first, then programmatic <critique>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver import (
    classify_next_steps,
    find_winning_paths,
    get_possible_moves,
    is_solvable,
    Step,
)
from prm_format import (
    build_prm_chatml_prompt,
    build_prm_completion,
)

CSV_CACHE_PATH = Path(__file__).resolve().parent / "data" / "24.csv"


def load_puzzles(csv_path: Path = CSV_CACHE_PATH) -> list[dict[str, Any]]:
    """Loads puzzles from 24.csv."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        puzzles = []
        for row in reader:
            nums = [int(x) for x in row["Puzzles"].split()]
            puzzles.append({
                "rank": int(row["Rank"]),
                "numbers": nums,
                "raw": row["Puzzles"],
            })
    return puzzles


def download_or_load_csv(cache_path: str | Path = CSV_CACHE_PATH) -> list[dict[str, Any]]:
    """Backwards-compatible alias for load_puzzles."""
    return load_puzzles(Path(cache_path))


def make_oracle_critique(
    candidate_step: str,
    label: float,
    error_type: str | None,
    remaining: tuple[Fraction, ...],
    meta: dict[str, Any] | None = None,
) -> str:
    """Generates 100% deterministic, accurate, hallucination-free critique from the oracle."""
    rem_str = [str(x) for x in remaining]

    if label == 1.0:
        if len(remaining) == 1 and remaining[0] == Fraction(24, 1):
            return f"Step '{candidate_step}' is sound and terminates exactly at target 24."
        hint = meta.get("winning_step_hint", "") if meta else ""
        if hint:
            return (
                f"Arithmetic is sound. Remaining numbers {rem_str} can reach 24 "
                f"(e.g., via downstream move '{hint}')."
            )
        return f"Arithmetic is sound. Remaining numbers {rem_str} preserve solvability to target 24."
    else:
        if error_type == "arithmetic_error":
            detail = meta.get("arithmetic_detail", "") if meta else ""
            return f"Fatal arithmetic error in '{candidate_step}': {detail}. The calculated result is invalid."
        elif error_type == "terminal_deviation":
            final_val = meta.get("final_val", rem_str[0] if rem_str else "?") if meta else ""
            return f"Step produces final value {final_val}, which does not reach target 24."
        else:  # dead_end_trap
            if len(remaining) == 2:
                return (
                    f"Arithmetic is correct, but remaining numbers {rem_str} cannot reach 24 under "
                    f"any arithmetic operation (+, -, *, /)."
                )
            return (
                f"Arithmetic is correct, but remaining state {rem_str} is unsolvable. "
                f"No sequence of standard operations on these numbers can produce 24."
            )


def state_hash(state_tuple: tuple[Fraction, ...]) -> int:
    """Computes a stable deterministic hash for a state multiset."""
    key = ",".join(str(x) for x in sorted(state_tuple))
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)


def generate_records_for_puzzle(
    puzzle_nums: list[int],
    puzzle_id: str,
) -> list[dict[str, Any]]:
    """Generates step-level records for a single puzzle across legal and mutated trajectories."""
    init_fractions = tuple(Fraction(x, 1) for x in puzzle_nums)
    solvable = is_solvable(init_fractions)
    if not solvable:
        return []

    winning_paths = find_winning_paths(init_fractions, max_paths=3)
    if not winning_paths:
        return []

    records: list[dict[str, Any]] = []

    for path_idx, path in enumerate(winning_paths):
        # Walk down this winning path and branch out at each step
        curr_state = init_fractions
        history: list[str] = []

        for step_idx, step in enumerate(path):
            step_num = step_idx + 1

            # 1. Positive move from winning path
            winning_step_hint = path[step_idx + 1].description() if step_idx + 1 < len(path) else ""
            critique = make_oracle_critique(
                step.description(),
                label=1.0,
                error_type=None,
                remaining=step.remaining,
                meta={"winning_step_hint": winning_step_hint},
            )
            prompt = build_prm_chatml_prompt(puzzle_nums, history, step.description())
            completion = build_prm_completion(1.0, critique)

            records.append({
                "prompt": prompt,
                "completion": completion,
                "label": 1.0,
                "error_type": None,
                "n_numbers": len(step.remaining),
                "state": [str(x) for x in sorted(step.remaining)],
                "puzzle_id": puzzle_id,
            })

            # 2. Branch out: check alternative candidate steps from curr_state
            alt_winning, alt_dead = classify_next_steps(curr_state)

            # Sample dead-end steps
            for dead_step in alt_dead[:3]:
                d_critique = make_oracle_critique(
                    dead_step.description(),
                    label=0.0,
                    error_type="dead_end_trap",
                    remaining=dead_step.remaining,
                )
                d_prompt = build_prm_chatml_prompt(puzzle_nums, history, dead_step.description())
                d_completion = build_prm_completion(0.0, d_critique)
                records.append({
                    "prompt": d_prompt,
                    "completion": d_completion,
                    "label": 0.0,
                    "error_type": "dead_end_trap",
                    "n_numbers": len(dead_step.remaining),
                    "state": [str(x) for x in sorted(dead_step.remaining)],
                    "puzzle_id": puzzle_id,
                })

            # 3. Inject synthetic arithmetic mutations at this step
            if len(curr_state) >= 2:
                n1, n2 = curr_state[0], curr_state[1]
                op = random.choice(["+", "-", "*"])
                if op == "+":
                    exp = n1 + n2
                    fake = exp + random.choice([1, -1, 2])
                elif op == "-":
                    exp = n1 - n2
                    fake = exp + random.choice([1, -1, 2])
                else:
                    exp = n1 * n2
                    fake = exp + random.choice([1, -1])

                bad_step_desc = f"{n1} {op} {n2} = {fake}"
                detail = f"{n1} {op} {n2} is {exp}, not {fake}"
                a_critique = make_oracle_critique(
                    bad_step_desc,
                    label=0.0,
                    error_type="arithmetic_error",
                    remaining=curr_state,
                    meta={"arithmetic_detail": detail},
                )
                a_prompt = build_prm_chatml_prompt(puzzle_nums, history, bad_step_desc)
                a_completion = build_prm_completion(0.0, a_critique)
                records.append({
                    "prompt": a_prompt,
                    "completion": a_completion,
                    "label": 0.0,
                    "error_type": "arithmetic_error",
                    "n_numbers": len(curr_state),
                    "state": [str(x) for x in sorted(curr_state)],
                    "puzzle_id": puzzle_id,
                })

            # Advance along the primary path
            history.append(f"Step {step_num}: {step.description()}")
            curr_state = step.remaining

            # 4. Check terminal deviation if length becomes 1
            if len(curr_state) == 1 and curr_state[0] != Fraction(24, 1):
                t_critique = make_oracle_critique(
                    step.description(),
                    label=0.0,
                    error_type="terminal_deviation",
                    remaining=curr_state,
                    meta={"final_val": str(curr_state[0])},
                )
                t_prompt = build_prm_chatml_prompt(puzzle_nums, history[:-1], step.description())
                t_completion = build_prm_completion(0.0, t_critique)
                records.append({
                    "prompt": t_prompt,
                    "completion": t_completion,
                    "label": 0.0,
                    "error_type": "terminal_deviation",
                    "n_numbers": 1,
                    "state": [str(curr_state[0])],
                    "puzzle_id": puzzle_id,
                })

    return records


def main():
    parser = argparse.ArgumentParser(description="Generate PRM v2 Oracle-Labelled Dataset")
    parser.add_argument("--out-dir", type=str, default="prm_data_v2", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target-train", type=int, default=20000, help="Target balanced train rows")
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PRM v2 DATASET GENERATOR".center(60))
    print("=" * 60)

    puzzles = load_puzzles()
    print(f"Loaded {len(puzzles)} puzzles from 24.csv")

    # Benchmark puzzles partition: ranks 901-950
    bench_puzzles = [p for p in puzzles if 901 <= p["rank"] <= 950]
    source_puzzles = [p for p in puzzles if not (901 <= p["rank"] <= 950)]
    print(f"Excluded {len(bench_puzzles)} benchmark puzzles (ranks 901-950) from training source.")
    print(f"Source training puzzles available: {len(source_puzzles)}")

    # 1. Enumerate benchmark test states
    print("\n[BENCH] Generating benchmark evaluation states from ranks 901-950...")
    bench_records: list[dict[str, Any]] = []
    bench_states_set: set[str] = set()
    for bp in bench_puzzles:
        b_recs = generate_records_for_puzzle(bp["numbers"], f"tot_{bp['rank']:04d}")
        for r in b_recs:
            st_key = ",".join(r["state"])
            bench_states_set.add(st_key)
            bench_records.append(r)

    print(f"[BENCH] Generated {len(bench_records)} records across {len(bench_states_set)} unique states.")

    # 2. Enumerate source training records
    print("\n[SOURCE] Generating records from source puzzles...")
    all_source_records: list[dict[str, Any]] = []
    for idx, sp in enumerate(source_puzzles):
        s_recs = generate_records_for_puzzle(sp["numbers"], f"tot_{sp['rank']:04d}")
        all_source_records.extend(s_recs)
        if (idx + 1) % 200 == 0 or (idx + 1) == len(source_puzzles):
            print(f"  Processed {idx + 1}/{len(source_puzzles)} source puzzles -> {len(all_source_records)} raw records.")

    # Add unsolvable random 4-tuples for root-level negative coverage
    print("\n[NEGATIVE] Generating unsolvable random 4-tuples...")
    unsolvable_count = 0
    attempts = 0
    while unsolvable_count < 400 and attempts < 3000:
        attempts += 1
        nums = [random.randint(1, 13) for _ in range(4)]
        fracs = tuple(Fraction(x, 1) for x in nums)
        if not is_solvable(fracs):
            unsolvable_count += 1
            # generate dead end step 1 moves
            moves = get_possible_moves(fracs)
            for m in moves[:3]:
                cand_str = f"Step 1: {m.description()}"
                critique = make_oracle_critique(cand_str, label=0.0, error_type="dead_end_trap", remaining=m.remaining)
                prompt = build_prm_chatml_prompt(nums, [], cand_str)
                comp = build_prm_completion(0.0, critique)
                all_source_records.append({
                    "prompt": prompt,
                    "completion": comp,
                    "label": 0.0,
                    "error_type": "dead_end_trap",
                    "n_numbers": len(m.remaining),
                    "state": [str(x) for x in sorted(m.remaining)],
                    "puzzle_id": f"unsolv_{unsolvable_count:04d}",
                })

    print(f"[NEGATIVE] Added {unsolvable_count} unsolvable puzzles.")
    print(f"[TOTAL] Total raw source candidate records: {len(all_source_records)}")

    # 3. Filter out any records that share states with benchmark set (LEAKAGE GUARD)
    leakage_count = 0
    clean_records: list[dict[str, Any]] = []
    for r in all_source_records:
        st_key = ",".join(r["state"])
        if st_key in bench_states_set:
            leakage_count += 1
        else:
            clean_records.append(r)
    print(f"[LEAKAGE GUARD] Filtered out {leakage_count} source records that overlapped with benchmark states.")
    print(f"[CLEAN] Clean training pool: {len(clean_records)} records.")

    # 4. Disjoint State Hash Split: 90% train / 5% valid / 5% test
    train_pool: list[dict[str, Any]] = []
    valid_pool: list[dict[str, Any]] = []
    test_pool: list[dict[str, Any]] = []

    for r in clean_records:
        st_tuple = tuple(Fraction(x) for x in r["state"])
        h = state_hash(st_tuple) % 100
        if h < 90:
            train_pool.append(r)
        elif h < 95:
            valid_pool.append(r)
        else:
            test_pool.append(r)

    print(f"\n[SPLIT] State-hashed pool sizes: train={len(train_pool)}, valid={len(valid_pool)}, test={len(test_pool)}")

    # 5. Balance Train 50/50
    pos_train = [r for r in train_pool if r["label"] == 1.0]
    neg_train = [r for r in train_pool if r["label"] == 0.0]
    print(f"[TRAIN POOL] Positive: {len(pos_train)}, Negative: {len(neg_train)}")

    target_per_class = min(len(pos_train), len(neg_train), args.target_train // 2)
    random.shuffle(pos_train)
    random.shuffle(neg_train)
    balanced_train = pos_train[:target_per_class] + neg_train[:target_per_class]
    random.shuffle(balanced_train)
    print(f"[BALANCED TRAIN] Final train size: {len(balanced_train)} (50% positive / 50% negative)")

    # 6. Save files
    def save_jsonl(records: list[dict[str, Any]], filename: str):
        path = out_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(records)} records to {path}")

    save_jsonl(balanced_train, "train.jsonl")
    save_jsonl(valid_pool, "valid.jsonl")
    save_jsonl(test_pool, "test.jsonl")
    save_jsonl(bench_records, "bench_states.jsonl")

    # 7. Token check for Qwen tokenizer
    meta = {
        "train_samples": len(balanced_train),
        "valid_samples": len(valid_pool),
        "test_samples": len(test_pool),
        "bench_samples": len(bench_records),
        "train_class_balance": {
            "pos": sum(1 for r in balanced_train if r["label"] == 1.0),
            "neg": sum(1 for r in balanced_train if r["label"] == 0.0),
        },
        "score_token_ids": {
            "1": 16,  # Standard Qwen token id for '1'
            "0": 15,  # Standard Qwen token id for '0'
        },
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✅ Dataset generation complete in {out_dir}/".center(60))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
