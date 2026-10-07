#!/usr/bin/env python3
"""
train_prm.py — MLX LoRA Fine-Tuning for the Process Reward Model (PRM)
Runs on Apple Silicon using mlx-lm-lora.
Includes real-time output streaming, logging, and an active NaN Watchdog to prevent
diverged runs from burning compute and poisoning checkpoint weights.
"""

import os
import sys
import subprocess
import yaml
import re
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
    RED = "\033[91m"
    GREEN = "\033[92m"
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


# Regex to detect NaN or Inf loss in training output
NAN_PATTERN = re.compile(
    r"(?:\b(?:val\s+)?loss\b|['\"]loss['\"])\s*[:=]?\s*['\"]?(?:[-+]?(?:nan|inf))\b",
    re.IGNORECASE,
)


def run_training_with_watchdog(cmd, env=None, log_file=None, watchdog_enabled=True):
    """
    Execute training subprocess while streaming stdout/stderr in real time.
    If watchdog_enabled is True, immediately aborts process upon detecting NaN/Inf loss.
    """
    if env is None:
        env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_handle = None
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "a", encoding="utf-8")

    if watchdog_enabled:
        print(f"{Colors.GREEN}[WATCHDOG]{Colors.RESET} Active monitoring enabled: will immediately abort if NaN/Inf loss is detected.\n")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    nan_detected = False
    trigger_line = ""

    try:
        buffer = ""
        while True:
            chunk = proc.stdout.read(64)
            if not chunk:
                if buffer:
                    sys.stdout.write(buffer)
                    sys.stdout.flush()
                    if log_handle:
                        log_handle.write(buffer)
                        log_handle.flush()
                break

            buffer += chunk
            while "\n" in buffer or "\r" in buffer:
                idx_n = buffer.find("\n")
                idx_r = buffer.find("\r")
                if idx_n != -1 and idx_r != -1:
                    idx = min(idx_n, idx_r)
                elif idx_n != -1:
                    idx = idx_n
                else:
                    idx = idx_r

                line = buffer[: idx + 1]
                buffer = buffer[idx + 1 :]

                sys.stdout.write(line)
                sys.stdout.flush()
                if log_handle:
                    log_handle.write(line)
                    log_handle.flush()

                if watchdog_enabled and NAN_PATTERN.search(line):
                    nan_detected = True
                    trigger_line = line.strip()
                    break

            if nan_detected:
                break

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}[WATCHDOG] KeyboardInterrupt received. Terminating training (PID {proc.pid})...{Colors.RESET}")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise

    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        if log_handle:
            log_handle.close()

    if nan_detected:
        print(f"\n{Colors.RED}{'═'*70}{Colors.RESET}")
        print(f"{Colors.RED}{Colors.BOLD}╔═════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.RED}{Colors.BOLD}║                   [WATCHDOG CRITICAL ALERT]                         ║{Colors.RESET}")
        print(f"{Colors.RED}{Colors.BOLD}║   NaN / Inf loss detected! Aborting training process immediately!   ║{Colors.RESET}")
        print(f"{Colors.RED}{Colors.BOLD}╚═════════════════════════════════════════════════════════════════════╝{Colors.RESET}")
        print(f"{Colors.YELLOW}Trigger output: {trigger_line}{Colors.RESET}")
        print(f"{Colors.RED}[WATCHDOG] Terminating training PID {proc.pid} to protect compute & weights...{Colors.RESET}")
        print(f"{Colors.RED}{'═'*70}{Colors.RESET}\n")

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

        sys.exit(2)

    return_code = proc.wait()
    if return_code != 0:
        print(f"\nTraining failed with exit code {return_code}")
        sys.exit(return_code)

    return return_code


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PRM LoRA Fine-Tuning Launcher")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Path to training config YAML")
    parser.add_argument("--adapter-path", type=str, default=None, help="Override adapter save path")
    parser.add_argument("--log-file", type=str, default=None, help="Save training logs to file")
    parser.add_argument("--no-watchdog", action="store_true", help="Disable NaN watchdog monitoring")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without running training")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"  # Use local cache only; avoids indefinite hang on repo_info TCP connect

    config_file = Path(args.config).resolve()
    if not config_file.exists():
        print(f"Error: Config file not found at {config_file}")
        sys.exit(1)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if args.adapter_path:
        config["adapter_path"] = args.adapter_path

    data_dir = config_file.parent / config.get("data", "prm_data")
    train_file = data_dir / "train.jsonl"

    if not train_file.exists():
        print(f"Dataset not found in {data_dir}. Generating now...")
        gen_script = REPO_ROOT / "mcts_reasoner" / "generate_prm_dataset.py"
        subprocess.run([sys.executable, str(gen_script)], check=True)

    # Check for --dry-run
    if args.dry_run:
        print_banner()
        print_section("Configuration Summary")
        print(f"{Colors.WHITE}Config:{Colors.RESET} {config_file}")
        print(f"{Colors.WHITE}Model:{Colors.RESET} {config.get('model')}")
        print(f"{Colors.WHITE}Training Mode:{Colors.RESET} SFT")
        print(f"{Colors.WHITE}Training Type:{Colors.RESET} lora")
        print(f"{Colors.WHITE}Data:{Colors.RESET} {config.get('data')}")
        print(f"{Colors.WHITE}Batch Size:{Colors.RESET} {config.get('batch_size')}")
        print(f"{Colors.WHITE}Iterations:{Colors.RESET} {config.get('iters')}")
        print(f"{Colors.WHITE}Learning Rate:{Colors.RESET} {config.get('learning_rate')}")
        print(f"{Colors.WHITE}Optimizer:{Colors.RESET} {config.get('optimizer', 'adamw')}")
        print(f"{Colors.WHITE}Gradient Accumulation:{Colors.RESET} {config.get('gradient_accumulation_steps', 1)}")
        print(f"{Colors.WHITE}Adapter Path:{Colors.RESET} {config.get('adapter_path')}")
        print(f"{Colors.WHITE}Watchdog:{Colors.RESET} {'Disabled' if args.no_watchdog else 'Enabled (Active)'}")
        print("\n[INFO] Dry-run complete. Training not started.")
        return

    # Determine backend: prefer mlx-lm-lora (has banner built-in)
    has_mlx_lm_lora = False
    try:
        import mlx_lm_lora  # noqa: F401
        has_mlx_lm_lora = True
    except ImportError:
        pass

    grad_accum = str(config.get("gradient_accumulation_steps", 1))
    optimizer = str(config.get("optimizer", "adamw"))
    adapter_path_arg = config.get("adapter_path", "prm_adapters_v2")

    if has_mlx_lm_lora:
        # mlx_lm_lora automatically prints the banner and configuration summary
        cmd = [
            sys.executable, "-m", "mlx_lm_lora", "train",
            "--config", str(config_file),
            "--adapter-path", adapter_path_arg,
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
            "--config", str(config_file),
            "--adapter-path", adapter_path_arg,
            "--grad-accumulation-steps", grad_accum,
        ]

    print(f"[LAUNCH] {' '.join(cmd)}\n")
    run_training_with_watchdog(
        cmd,
        log_file=args.log_file,
        watchdog_enabled=not args.no_watchdog,
    )


if __name__ == "__main__":
    main()
