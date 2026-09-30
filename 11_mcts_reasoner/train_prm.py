#!/usr/bin/env python3
"""
train_prm.py — MLX LoRA Fine-Tuning for the Process Reward Model (PRM)
Runs on Apple Silicon using mlx-lm-lora (matching dpo_reviewer).
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "prm_config.yaml"

# ANSI Colors matching dpo_reviewer / mlx-lm-lora visuals
class Colors:
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[35m"
    BLUE = "\033[94m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_banner():
    """Print the authentic MLX LM LORA ASCII banner matching dpo_reviewer."""
    banner = f"""
{Colors.CYAN}╔═════════════════════════════════════════════════════════════════════════════════════════════════╗{Colors.RESET}
{Colors.CYAN}║{Colors.RESET}                                                                                                 {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET} {Colors.BOLD}{Colors.MAGENTA}███╗   ███╗██╗     ██╗  ██╗    ██╗     ███╗   ███╗    ██╗      ██████╗ ██████╗  █████╗{Colors.RESET}          {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET} {Colors.BOLD}{Colors.MAGENTA}████╗ ████║██║     ╚██╗██╔╝    ██║     ████╗ ████║    ██║     ██╔═══██╗██╔══██╗██╔══██╗{Colors.RESET}         {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET} {Colors.BOLD}{Colors.BLUE}██╔████╔██║██║      ╚███╔╝     ██║     ██╔████╔██║    ██║     ██║   ██║██████╔╝███████║{Colors.RESET}         {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET} {Colors.BOLD}{Colors.BLUE}██║╚██╔╝██║██║      ██╔██╗     ██║     ██║╚██╔╝██║    ██║     ██║   ██║██╔══██╗██╔══██║{Colors.RESET}         {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET} {Colors.BOLD}{Colors.CYAN}██║ ╚═╝ ██║███████╗██╔╝ ██╗    ███████╗██║ ╚═╝ ██║    ███████╗╚██████╔╝██║  ██║██║  ██║{Colors.RESET}         {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET} {Colors.BOLD}{Colors.CYAN}╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝    ╚══════╝╚═╝     ╚═╝    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝{Colors.RESET}         {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET}                                                                                                 {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET} {Colors.YELLOW}{Colors.BOLD}Advanced Fine-Tuning Framework{Colors.RESET}                                                                  {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET} {Colors.DIM}{Colors.WHITE}LoRA • (Online-)DPO • XPO • CPO • CPO • ORPO • PPO • FTPO • GRPO • DrGRPO • GSPO • RLHF • SFT{Colors.RESET}   {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET}                                                                                                 {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}╚═════════════════════════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.WHITE}{title.center(60)}{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"  # Use local cache only; avoids indefinite hang on repo_info TCP connect

    if not CONFIG_PATH.exists():
        print(f"Error: Config file not found at {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    data_dir = CONFIG_PATH.parent / config.get("data", "prm_data")
    train_file = data_dir / "train.jsonl"

    if not train_file.exists():
        print(f"Dataset not found in {data_dir}. Generating now...")
        gen_script = REPO_ROOT / "mcts_reasoner" / "generate_prm_dataset.py"
        subprocess.run([sys.executable, str(gen_script)], check=True)

    # Check for --dry-run
    if "--dry-run" in sys.argv:
        print_banner()
        print_section("Configuration Summary")
        print(f"{Colors.WHITE}Model:{Colors.RESET} {config.get('model')}")
        print(f"{Colors.WHITE}Training Mode:{Colors.RESET} SFT")
        print(f"{Colors.WHITE}Training Type:{Colors.RESET} lora")
        print(f"{Colors.WHITE}Batch Size:{Colors.RESET} {config.get('batch_size')}")
        print(f"{Colors.WHITE}Learning Rate:{Colors.RESET} {config.get('learning_rate')}")
        print(f"{Colors.WHITE}Optimizer:{Colors.RESET} {config.get('optimizer', 'adamw')}")
        print(f"{Colors.WHITE}Gradient Accumulation:{Colors.RESET} {config.get('gradient_accumulation_steps', 4)}")
        print("\n[INFO] Dry-run complete. Training not started.")
        return

    # Determine backend: prefer mlx-lm-lora (has banner built-in)
    has_mlx_lm_lora = False
    try:
        import mlx_lm_lora  # noqa: F401
        has_mlx_lm_lora = True
    except ImportError:
        pass

    grad_accum = str(config.get("gradient_accumulation_steps", 4))
    optimizer = str(config.get("optimizer", "adamw"))

    if has_mlx_lm_lora:
        # mlx_lm_lora automatically prints the banner and configuration summary
        cmd = [
            sys.executable, "-m", "mlx_lm_lora", "train",
            "--config", str(CONFIG_PATH),
            "--gradient-accumulation-steps", grad_accum,
            "--optimizer", optimizer,
        ]
    else:
        # Fallback to standard mlx_lm.lora: print banner & summary ourselves first
        print_banner()
        print_section("Configuration Summary")
        print(f"{Colors.WHITE}Model:{Colors.RESET} {config.get('model')}")
        print(f"{Colors.WHITE}Training Mode:{Colors.RESET} SFT")
        print(f"{Colors.WHITE}Training Type:{Colors.RESET} lora")
        print(f"{Colors.WHITE}Batch Size:{Colors.RESET} {config.get('batch_size')}")
        print(f"{Colors.WHITE}Learning Rate:{Colors.RESET} {config.get('learning_rate')}")
        print(f"{Colors.WHITE}Optimizer:{Colors.RESET} {optimizer}")
        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--config", str(CONFIG_PATH),
            "--grad-accumulation-steps", grad_accum,
        ]

    print(f"[LAUNCH] {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()

