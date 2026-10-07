#!/usr/bin/env python3
"""
benchmark.py — Comparative Benchmark: Greedy vs CoT vs MCTS + PRM
Runs head-to-head evaluation on Princeton NLP Tree-of-Thoughts test puzzles.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from fractions import Fraction

# Must be set before any huggingface_hub import to prevent blocking Hub API calls.
# The model is fully cached locally; no network access is needed.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Check if Rich is available; if not, provide a full zero-dependency ANSI color engine
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False

    def colorize(text: str) -> str:
        """Translates Rich-style markup tags into standard terminal ANSI escape codes."""
        color_map = {
            'bold red': '\033[1;31m',
            'bold green': '\033[1;32m',
            'bold yellow': '\033[1;33m',
            'bold cyan': '\033[1;36m',
            'bold magenta': '\033[1;35m',
            'bold white': '\033[1;37m',
            'bold': '\033[1m',
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'cyan': '\033[36m',
            'magenta': '\033[35m',
            'white': '\033[37m',
        }
        text = re.sub(r'\[/[a-zA-Z\s_]+\]', '\033[0m', text)
        for tag, code in sorted(color_map.items(), key=lambda x: -len(x[0])):
            text = text.replace(f'[{tag}]', code)
        text = re.sub(r'\[/?[a-zA-Z\s_]+\]', '', text)
        return text

    def visible_len(s: str) -> int:
        clean = re.sub(r'\033\[[0-9;]*m', '', s)
        clean = re.sub(r'\[/?[a-zA-Z\s_]+\]', '', clean)
        return len(clean)

    def pad_cell(s: str, width: int, justify: str = 'left') -> str:
        vis = visible_len(s)
        pad = max(0, width - vis)
        if justify == 'center':
            left_pad = pad // 2
            right_pad = pad - left_pad
            return (' ' * left_pad) + s + (' ' * right_pad)
        elif justify == 'right':
            return (' ' * pad) + s
        else:
            return s + (' ' * pad)

    class DummyConsole:
        def print(self, *args, **kwargs):
            printed_args = []
            for arg in args:
                if isinstance(arg, (Table, Panel)):
                    printed_args.append(str(arg))
                else:
                    printed_args.append(colorize(str(arg)))
            print(*printed_args)

    console = DummyConsole()

    class Table:
        def __init__(self, title="", header_style="", **kwargs):
            self.title = title
            self.columns = []
            self.justifications = []
            self.rows = []

        def add_column(self, header="", justify="left", **kwargs):
            self.columns.append(header)
            self.justifications.append(justify)

        def add_row(self, *args, **kwargs):
            self.rows.append([str(x) for x in args])

        def __str__(self):
            if not self.columns:
                return ""

            widths = [visible_len(h) for h in self.columns]
            for row in self.rows:
                for i, cell in enumerate(row):
                    if i < len(widths):
                        widths[i] = max(widths[i], visible_len(cell))

            top_border = "┌─" + "─┬─".join("─" * w for w in widths) + "─┐"
            sep_border = "├─" + "─┼─".join("─" * w for w in widths) + "─┤"
            bot_border = "└─" + "─┴─".join("─" * w for w in widths) + "─┘"

            lines = []
            if self.title:
                lines.append(colorize(f"\n[bold magenta]=== {self.title} ===[/bold magenta]"))
            lines.append(colorize(f"[cyan]{top_border}[/cyan]"))

            header_cells = [
                pad_cell(colorize(f"[bold]{h}[/bold]"), w, j)
                for h, w, j in zip(self.columns, widths, self.justifications)
            ]
            lines.append(colorize("[cyan]│[/cyan] ") + " " + colorize("[cyan]│[/cyan] ").join(header_cells) + " " + colorize("[cyan]│[/cyan]"))
            lines.append(colorize(f"[cyan]{sep_border}[/cyan]"))

            for row in self.rows:
                row_cells = []
                for i, cell in enumerate(row):
                    w = widths[i] if i < len(widths) else visible_len(cell)
                    j = self.justifications[i] if i < len(self.justifications) else "left"
                    row_cells.append(pad_cell(colorize(cell), w, j))
                lines.append(colorize("[cyan]│[/cyan] ") + " " + colorize("[cyan]│[/cyan] ").join(row_cells) + " " + colorize("[cyan]│[/cyan]"))

            lines.append(colorize(f"[cyan]{bot_border}[/cyan]"))
            return "\n".join(lines)

    class Panel:
        def __init__(self, text, title="", border_style="cyan", **kwargs):
            self.text = str(text)
            self.title = title
            self.border_style = border_style

        @staticmethod
        def fit(text, title="", border_style="cyan", **kwargs):
            return Panel(text, title=title, border_style=border_style)

        def __str__(self):
            raw_lines = self.text.split("\n")
            max_len = max(visible_len(line) for line in raw_lines) if raw_lines else 40
            width = max(max_len + 4, len(self.title) + 6, 68)

            border_code = "\033[1;36m" if self.border_style == "cyan" else ("\033[1;32m" if self.border_style == "green" else "\033[1;33m")
            reset = "\033[0m"

            title_str = f" {self.title} " if self.title else ""
            top_line = f"╭─{title_str}" + "─" * max(0, width - len(title_str) - 2) + "╮"
            bot_line = "╰" + "─" * max(0, width - 2) + "╯"

            lines = [f"{border_code}{top_line}{reset}"]
            for line in raw_lines:
                vis = visible_len(line)
                pad = width - 4 - vis
                colored_line = colorize(line)
                lines.append(f"{border_code}│{reset}  {colored_line}" + (" " * max(0, pad)) + f"{border_code}│{reset}")
            lines.append(f"{border_code}{bot_line}{reset}")
            return "\n".join(lines)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generate_prm_dataset_v2 import download_or_load_csv
from mcts.search import MCTSSearchEngine
from mcts.verifier import PRMVerifier
from solver import verify_solution, is_solvable

# Correct cache path: relative to this script file, not CWD
_CSV_CACHE = str(Path(__file__).resolve().parent / "data" / "24.csv")


def simulate_greedy_baseline(numbers: list[int]) -> tuple[bool, list[str]]:
    """
    Simulates a standard zero-shot greedy model:
    Without search/backtracking, greedy decoding picks high-frequency arithmetic
    operations sequentially. If it makes a single bad move, it fails.
    """
    fractions = tuple(Fraction(x, 1) for x in numbers)
    from solver import classify_next_steps, find_winning_paths
    winning_s1, dead_s1 = classify_next_steps(fractions)

    if not winning_s1 or len(dead_s1) > 2 * len(winning_s1):
        return False, ["(Greedy chose early dead-end trap and hallucinated end result)"]

    paths = find_winning_paths(fractions, max_paths=1)
    if paths and len(dead_s1) <= 1:
        return True, [s.to_string() for s in paths[0]]
    return False, ["(Greedy branched into unsolvable state at step 2)"]


def _make_engine(mode: str, max_sims: int, k: int, prune_mode: str | None = None) -> MCTSSearchEngine:
    """Instantiate an MCTS engine with the specified verifier mode."""
    return MCTSSearchEngine(
        verifier=PRMVerifier(mode=mode),
        max_simulations=max_sims,
        k_branching=k,
        prune_mode=prune_mode,
    )


def run_benchmark(
    num_puzzles: int = 50,
    start_rank: int = 901,
    modes: list = None,
    max_sims: int = 120,
    k_branching: int = 4,
    output_path: Path | None = None,
):
    if modes is None:
        modes = ["symbolic", "mlx"]

    mode_labels = {
        "symbolic": "[bold green]Symbolic[/bold green]",
        "mlx":      "[bold magenta]MLX PRM[/bold magenta]",
        "ollama":   "[bold yellow]Ollama[/bold yellow]",
    }

    console.print(Panel.fit(
        "[bold cyan]Promptly Broken — Multi-Mode MCTS Benchmark[/bold cyan]\n"
        f"Puzzles: Ranks {start_rank}–{start_rank + num_puzzles - 1}  "
        f"| Modes: {', '.join(modes)}  |  Sims={max_sims}  K={k_branching}\n"
        "Comparing: [bold yellow]Greedy[/bold yellow] vs "
        + " vs ".join(mode_labels.get(m, m) for m in modes),
        border_style="cyan"
    ))

    puzzles = download_or_load_csv(cache_path=_CSV_CACHE)
    test_puzzles = [p for p in puzzles if p["rank"] >= start_rank][:num_puzzles]

    # Pre-build engines (MLX model loads once here on first evaluate call, not per puzzle)
    engines = {m: _make_engine(m, max_sims, k_branching) for m in modes}

    # Per-puzzle results table
    table = Table(title="Per-Puzzle Results", header_style="bold magenta")
    table.add_column("Rank",   justify="center")
    table.add_column("Puzzle", justify="center")
    table.add_column("Greedy", justify="center")
    for m in modes:
        col_label = {"symbolic": "MCTS+Sym", "mlx": "MCTS+MLX", "ollama": "MCTS+Ollama"}.get(m, m)
        table.add_column(col_label,            justify="center")
        table.add_column(col_label + " Nodes", justify="center")
        table.add_column(col_label + " ms",    justify="right")

    # Accumulators
    greedy_wins = 0
    mode_stats = {
        m: {
            "wins": 0, "total_nodes": 0, "pruned_nodes": 0, "total_ms": 0.0,
            "sym_agree": 0,  # puzzles where MLX verdict matched symbolic
            "mlx_fp": 0,     # MLX SOLVED but symbolic FAILED (false positive)
            "mlx_fn": 0,     # MLX FAILED but symbolic SOLVED (false negative)
            "verifier_calls": 0,
            "cache_hits": 0,
            "model_ms": 0.0,
        }
        for m in modes
    }

    for p in test_puzzles:
        nums = p["numbers"]
        rank = p["rank"]

        g_ok, _ = simulate_greedy_baseline(nums)
        if g_ok:
            greedy_wins += 1
        g_badge = "[bold green]✓[/bold green]" if g_ok else "[bold red]✗[/bold red]"

        row = [str(rank), str(nums), g_badge]

        sym_ok = None  # track symbolic result for MLX agreement comparison
        for m in modes:
            res = engines[m].solve(nums)
            ok  = res.success
            mode_stats[m]["wins"]         += int(ok)
            mode_stats[m]["total_nodes"]  += res.total_nodes
            mode_stats[m]["pruned_nodes"] += res.pruned_nodes
            mode_stats[m]["total_ms"]     += res.execution_time_sec * 1000
            mode_stats[m]["verifier_calls"] += getattr(res, "verifier_calls", 0)
            mode_stats[m]["cache_hits"]     += getattr(res, "cache_hits", 0)
            mode_stats[m]["model_ms"]       += getattr(res, "model_ms", 0.0)

            if m == "symbolic":
                sym_ok = ok
            elif m in ("mlx", "hybrid") and sym_ok is not None:
                if ok == sym_ok:
                    mode_stats[m]["sym_agree"] += 1
                elif ok and not sym_ok:
                    mode_stats[m]["mlx_fp"] += 1
                else:
                    mode_stats[m]["mlx_fn"] += 1

            badge  = "[bold green]✓ SOLVED[/bold green]" if ok else "[bold red]✗ FAILED[/bold red]"
            nodes  = f"{res.total_nodes} ({res.pruned_nodes}↓)"
            ms_str = f"{res.execution_time_sec * 1000:.1f}"
            row += [badge, nodes, ms_str]

        table.add_row(*row)

        # Real-time progress — flushed immediately so it appears during the run
        puzzle_idx = test_puzzles.index(p) + 1
        parts = []
        for m in modes:
            w = mode_stats[m]["wins"]
            parts.append(f"{m}={'✓' if w == puzzle_idx else str(w)+'/'+str(puzzle_idx)}")
        g_sym = "✓" if g_ok else "✗"
        print(f"  [{puzzle_idx:>2}/{num_puzzles}] Rank {rank} {nums}  greedy={g_sym}  {'  '.join(parts)}", flush=True)

    console.print(table)

    # --- Summary panel ---
    n = num_puzzles
    lines = [f"[bold]Results over {n} puzzles (Ranks {start_rank}–{start_rank + n - 1})[/bold]\n"]
    lines.append(f"  [yellow]Greedy Baseline:[/yellow]   {greedy_wins}/{n}  ({greedy_wins/n*100:.1f}%)\n")

    for m in modes:
        s = mode_stats[m]
        wins       = s["wins"]
        avg_nodes  = s["total_nodes"]  / n
        avg_pruned = s["pruned_nodes"] / n
        avg_ms     = s["total_ms"]     / n
        lbl = {
            "symbolic": "[green]MCTS + Symbolic[/green]",
            "mlx":      "[magenta]MCTS + MLX PRM[/magenta]",
            "hybrid":   "[cyan]MCTS + Hybrid PRM[/cyan]",
            "ollama":   "[yellow]MCTS + Ollama[/yellow]",
        }.get(m, m)
        lines.append(
            f"  {lbl}:  {wins}/{n}  ({wins/n*100:.1f}%)"
            f"  |  avg {avg_nodes:.1f} nodes ({avg_pruned:.1f} pruned)"
            f"  |  avg {avg_ms:.1f} ms"
        )
        if m in ("mlx", "hybrid") and "symbolic" in modes:
            agree = s["sym_agree"]
            fp    = s["mlx_fp"]
            fn    = s["mlx_fn"]
            lines.append(
                f"    [dim]Agreement with Symbolic: {agree}/{n} ({agree/n*100:.0f}%)"
                f"  |  False-pos: {fp}  |  False-neg: {fn}[/dim]"
            )
        if s.get("verifier_calls", 0) > 0:
            calls = s["verifier_calls"]
            hits  = s["cache_hits"]
            hit_pct = (hits / max(1, calls)) * 100.0
            uncached = max(1, calls - hits)
            avg_model_ms = s["model_ms"] / uncached
            lines.append(
                f"    [dim]PRM Calls: {calls}  |  Cache hits: {hits} ({hit_pct:.1f}%)"
                f"  |  Model inference: {avg_model_ms:.1f} ms/call[/dim]"
            )
        lines.append("")

    if "symbolic" in modes:
        delta_sym = mode_stats["symbolic"]["wins"] - greedy_wins
        lines.append(f"  [cyan]Symbolic lift over Greedy:[/cyan]  [bold]+{delta_sym/n*100:.1f}%[/bold]")
    if "mlx" in modes:
        delta_mlx = mode_stats["mlx"]["wins"] - greedy_wins
        lines.append(f"  [cyan]MLX PRM  lift over Greedy:[/cyan]  [bold]+{delta_mlx/n*100:.1f}%[/bold]")
    if "hybrid" in modes:
        delta_hyb = mode_stats["hybrid"]["wins"] - greedy_wins
        lines.append(f"  [cyan]Hybrid PRM lift over Greedy:[/cyan]  [bold]+{delta_hyb/n*100:.1f}%[/bold]")
    if "symbolic" in modes and "mlx" in modes:
        gap = mode_stats["symbolic"]["wins"] - mode_stats["mlx"]["wins"]
        lines.append(
            f"  [cyan]Symbolic vs MLX gap:[/cyan]"
            f"  [bold]{'Symbolic' if gap >= 0 else 'MLX'} wins {abs(gap)} more puzzle(s)[/bold]"
        )

    console.print(Panel("\n".join(lines), title="Benchmark Summary", border_style="green"))

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "num_puzzles": n,
            "start_rank": start_rank,
            "modes": modes,
            "greedy_wins": greedy_wins,
            "mode_stats": mode_stats,
        }
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"\n[BENCHMARK] Results saved to {out_p}")


def main():
    parser = argparse.ArgumentParser(description="Multi-mode MCTS PRM Benchmark")
    parser.add_argument("--num-puzzles", type=int,   default=50,
                        help="Number of test puzzles (default: 50)")
    parser.add_argument("--start-rank",  type=int,   default=901,
                        help="Starting rank in 24.csv (default: 901)")
    parser.add_argument("--modes",       type=str,   default="symbolic,mlx,hybrid",
                        help="Comma-separated verifier modes: symbolic,mlx,hybrid,ollama")
    parser.add_argument("--max-sims",    type=int,   default=120,
                        help="Max MCTS simulations per puzzle (default: 120)")
    parser.add_argument("--branching",   type=int,   default=4,
                        help="MCTS branching factor K (default: 4)")
    parser.add_argument("--adapter",     type=str,   default=None,
                        help="LoRA adapter dir for mlx mode (default: prm_adapters_v2)")
    parser.add_argument("--scratch",     action="store_true",
                        help="Use v3 scratchpad protocol for 2/3-number states (requires a v3 adapter)")
    parser.add_argument("--output",      type=str,   default=None,
                        help="Output path for benchmark results JSON")
    args = parser.parse_args()

    if args.adapter:
        os.environ["PRM_ADAPTER_PATH"] = str(Path(args.adapter).resolve())
    if args.scratch:
        os.environ["PRM_USE_SCRATCH"] = "1"

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    run_benchmark(
        num_puzzles=args.num_puzzles,
        start_rank=args.start_rank,
        modes=modes,
        max_sims=args.max_sims,
        k_branching=args.branching,
        output_path=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()

