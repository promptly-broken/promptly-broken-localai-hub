#!/usr/bin/env python3
"""
diag_v3.py — Diagnostics for the v3 scratchpad PRM (read-only; no training).

D2: generated scratchpad vs oracle scratchpad (exact / per-line), and does the verdict follow its own scratch?
D3: oracle scratchpad teacher-forced -> verdict-only accuracy (upper bound for the verdict step).
D4: tokenization boundary check between training strings and the inference prefill.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx
import numpy as np

from prm_format import VERDICT_PREFIX
from prm_infer import generate_scratchpad
from prm_scratch import SCRATCH_CLOSE, SCRATCH_OPEN


def oracle_scratch(row: dict) -> str:
    comp = row["completion"]
    return comp[: comp.index(SCRATCH_CLOSE) + len(SCRATCH_CLOSE)]


def verdict_prob(model, tok, text: str, one_id: int, zero_id: int) -> float:
    toks = mx.array(tok.encode(text))[None]
    logits = model(toks)[0, -1, :]
    pair = logits[mx.array([one_id, zero_id])].astype(mx.float32)
    return float(mx.softmax(pair)[0].item())


def parse_lines(scratch: str, n: int):
    """Returns list of dicts: {'vals': [...], 'hit': bool} per pair line (n=3) or one entry (n=2)."""
    body = scratch.replace(SCRATCH_OPEN, "").replace(SCRATCH_CLOSE, "").strip().split("\n")
    out = []
    try:
        if n == 2:
            vals = body[1].split(": ", 1)[1] if ": " in body[1] else body[0].split(": ", 1)[1]
            # layout: "[a, b]: v1, v2..." then hit/none
            first = body[0]
            vals = first.split(": ", 1)[1]
            hit = not body[1].strip().startswith("none")
            out.append({"vals": [v.strip() for v in vals.split(",")], "hit": hit})
        else:
            for line in body[1:4]:
                left, right = line.split("; ", 1)
                vals = left.split(": ", 1)[1]
                out.append({"vals": [v.strip() for v in vals.split(",")], "hit": not right.strip().startswith("none")})
    except Exception:
        return None
    return out


def reach_from_scratch(scratch: str, n: int) -> bool | None:
    p = parse_lines(scratch, n)
    if p is None:
        return None
    return any(x["hit"] for x in p)


def tokenization_checks(tok, rows) -> dict:
    """D4: does the inference-time prefill tokenise as a prefix of the training sequence?"""
    open_ok = close_ok = 0
    ex_open = ex_close = None
    for r in rows:
        full = r["prompt"] + r["completion"]
        ids_full = tok.encode(full)
        # (a) forced "<scratch>\n" prefill
        pre_open = tok.encode(r["prompt"] + SCRATCH_OPEN + "\n")
        ok_a = ids_full[: len(pre_open)] == pre_open
        open_ok += ok_a
        if not ok_a and ex_open is None:
            k = len(pre_open) - 1
            ex_open = {"prefill_tail": tok.decode(pre_open[-3:]), "training_tokens_at_boundary": [tok.decode([t]) for t in ids_full[k - 1:k + 3]]}
        # (b) scratch + "\n" + verdict prefix
        sc = oracle_scratch(r)
        pre_v = tok.encode(r["prompt"] + sc + "\n" + VERDICT_PREFIX)
        ok_b = ids_full[: len(pre_v)] == pre_v
        close_ok += ok_b
        if not ok_b and ex_close is None:
            ex_close = {"prefill_tail": [tok.decode([t]) for t in pre_v[-4:]], "training_tail": [tok.decode([t]) for t in ids_full[len(pre_v) - 4:len(pre_v) + 1]]}
    n = len(rows)
    return {"rows_checked": n, "open_prefill_matches_training": f"{open_ok}/{n}", "verdict_prefill_matches_training": f"{close_ok}/{n}",
            "example_open_mismatch": ex_open, "example_verdict_mismatch": ex_close}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="prm_data_v3/test.jsonl")
    ap.add_argument("--adapter", default="prm_adapters_v3")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("-n", "--per-size", type=int, default=75)
    ap.add_argument("--out", default="eval_results/v3_diag.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    random.seed(args.seed)
    sel = []
    for n in (2, 3):
        pool = [r for r in rows if r["n_numbers"] == n]
        random.shuffle(pool)
        sel += pool[: args.per_size]
    print(f"[DIAG] {len(sel)} rows ({args.per_size} per size) from {args.data}")

    from mlx_lm import load
    model, tok = load(args.model, adapter_path=args.adapter)
    one_id, zero_id = tok.encode("1")[0], tok.encode("0")[0]

    # ---- D4
    print("\n[D4] Tokenization boundary check ...")
    d4 = tokenization_checks(tok, sel[:60])
    print(json.dumps(d4, indent=2))

    # ---- D2 / D3
    stats = defaultdict(lambda: defaultdict(list))
    details = []
    t0 = time.time()
    for i, r in enumerate(sel):
        n = r["n_numbers"]
        label = r["label"]
        orc = oracle_scratch(r)
        gen, ntok, closed = generate_scratchpad(model, tok, r["prompt"])

        # D2: own scratch -> verdict
        p_own = verdict_prob(model, tok, r["prompt"] + gen + "\n" + VERDICT_PREFIX, one_id, zero_id)
        # D3: oracle scratch -> verdict
        p_orc = verdict_prob(model, tok, r["prompt"] + orc + "\n" + VERDICT_PREFIX, one_id, zero_id)

        po, pg = parse_lines(orc, n), parse_lines(gen, n)
        s = stats[n]
        s["exact"].append(gen == orc)
        s["closed"].append(closed)
        s["parsed"].append(pg is not None)
        if pg is not None and len(pg) == len(po):
            s["vals_list_match"].append(all(a["vals"] == b["vals"] for a, b in zip(pg, po)))
            s["vals_set_match"].append(all(set(a["vals"]) == set(b["vals"]) for a, b in zip(pg, po)))
            s["hit_match"].append(all(a["hit"] == b["hit"] for a, b in zip(pg, po)))
        else:
            s["vals_list_match"].append(False); s["vals_set_match"].append(False); s["hit_match"].append(False)
        reach_own = reach_from_scratch(gen, n)
        s["scratch_reach_correct"].append(reach_own == (label == 1.0) if reach_own is not None else False)
        if reach_own is not None:
            s["verdict_follows_scratch"].append((p_own >= 0.5) == reach_own)
        s["acc_own"].append((p_own >= 0.5) == (label == 1.0))
        s["acc_oracle_scratch"].append((p_orc >= 0.5) == (label == 1.0))
        s["label"].append(label); s["p_orc"].append(p_orc); s["p_own"].append(p_own)
        s["ntok"].append(ntok)

        details.append({"state": r["state"], "n": n, "label": label, "generated": gen, "oracle": orc,
                        "p_own": round(p_own, 4), "p_oracle_scratch": round(p_orc, 4), "closed": closed})
        if (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(sel)} ({time.time() - t0:.0f}s)")

    def rate(x):
        return round(float(np.mean(x)), 3) if len(x) else None

    def bal(lbl, p):
        lbl, p = np.array(lbl), np.array(p) >= 0.5
        pos, neg = lbl == 1.0, lbl == 0.0
        return round(float((p[pos].mean() + (~p[neg]).mean()) / 2), 3) if pos.any() and neg.any() else None

    summary = {"D4_tokenization": d4}
    for n in (2, 3):
        s = stats[n]
        ex = np.array(s["exact"])
        acc_own = np.array(s["acc_own"])
        summary[f"size_{n}"] = {
            "rows": len(ex),
            "pos_rows": int(sum(1 for l in s["label"] if l == 1.0)),
            "D2_scratch_exact_match": rate(ex),
            "D2_scratch_closed": rate(s["closed"]),
            "D2_values_list_match": rate(s["vals_list_match"]),
            "D2_values_set_match": rate(s["vals_set_match"]),
            "D2_hit_flags_match": rate(s["hit_match"]),
            "D2_scratch_conclusion_correct": rate(s["scratch_reach_correct"]),
            "D2_verdict_follows_own_scratch": rate(s["verdict_follows_scratch"]),
            "D2_end_to_end_acc": rate(s["acc_own"]),
            "D2_end_to_end_bal_acc": bal(s["label"], s["p_own"]),
            "D2_acc_when_scratch_exact": rate(acc_own[ex]) if ex.any() else None,
            "D2_acc_when_scratch_not_exact": rate(acc_own[~ex]) if (~ex).any() else None,
            "D3_oracle_scratch_acc": rate(s["acc_oracle_scratch"]),
            "D3_oracle_scratch_bal_acc": bal(s["label"], s["p_orc"]),
            "mean_scratch_tokens": round(float(np.mean(s["ntok"])), 1),
        }
    print("\n" + "=" * 70)
    print(json.dumps(summary, indent=2))
    print("=" * 70)

    # a few examples of wrong scratchpads for eyeballing
    wrong = [d for d in details if d["generated"] != d["oracle"]][:3]
    for d in wrong:
        print(f"\n--- EXAMPLE (n={d['n']} label={d['label']} state={d['state']}) ---")
        print("ORACLE:\n" + d["oracle"])
        print("GENERATED:\n" + d["generated"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "details": details}, open(out, "w"), indent=2)
    print(f"\n[DIAG] saved {out}")


if __name__ == "__main__":
    main()
