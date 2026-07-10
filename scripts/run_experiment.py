#!/usr/bin/env python3
"""
scripts/run_experiment.py

Orchestrates the entire experiment lifecycle for the DSA Tutor LLM:
1. Safety validation (checksums, space, config checks)
2. Environment metadata gathering for reproducibility
3. Executing training, benchmarking, and failure analysis
4. Compiling a final markdown report
Also supports comparison between experiments and resuming checkpointed runs.
"""

import os
import sys
import yaml
import json
import time
import shutil
import hashlib
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

# Setup paths to import scripts
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    import torch
    import transformers
    import peft
    import trl
except ImportError:
    pass

# Bypassing large integer string conversion limits
sys.set_int_max_str_digits(0)


def get_git_commit() -> str:
    """Safely obtain current git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "N/A"


def get_dataset_checksum(filepath: str) -> str:
    """Compute sha256 checksum of dataset file."""
    if not os.path.exists(filepath):
        return "missing"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_disk_space(directory: str, min_gb: float = 5.0):
    """Verify free disk space in directory path."""
    target = os.path.abspath(directory)
    while not os.path.exists(target):
        parent = os.path.dirname(target)
        if parent == target:
            break
        target = parent
    total, used, free = shutil.disk_usage(target)
    free_gb = free / (1024 ** 3)
    if free_gb < min_gb:
        raise RuntimeError(f"Safety Failure: Low disk space in {target}. Required: {min_gb} GB, Available: {free_gb:.2f} GB")


def gather_metadata(seed: int) -> dict:
    """Collect Python environment, library versions, GPU and system specs for reproducibility."""
    meta = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "library_versions": {
            "torch": torch.__version__ if "torch" in sys.modules else "unknown",
            "transformers": transformers.__version__ if "transformers" in sys.modules else "unknown",
            "peft": peft.__version__ if "peft" in sys.modules else "unknown",
            "trl": trl.__version__ if "trl" in sys.modules else "unknown",
        },
        "cuda_version": torch.version.cuda if (torch.cuda.is_available() if "torch" in sys.modules else False) else "N/A",
        "gpu_model": torch.cuda.get_device_name(0) if (torch.cuda.is_available() if "torch" in sys.modules else False) else "None",
        "random_seed": seed
    }
    return meta


def run_training_step(config_path: str, output_dir: str):
    """Executes train.py dynamically in-process."""
    import train
    with patch("sys.argv", ["train.py", "--config", config_path]):
        train.main()


def run_benchmarking_step(adapter_path: str):
    """Executes benchmark_base_model.py on adapter in-process."""
    import benchmark_base_model
    with patch("sys.argv", ["benchmark_base_model.py", "--adapter", adapter_path]):
        benchmark_base_model.main()


def run_failure_analysis_step():
    """Executes analyze_failures.py in-process comparing base vs adapter."""
    import analyze_failures
    with patch("sys.argv", ["analyze_failures.py"]):
        analyze_failures.main()


def run_comparison(exp_a: str, exp_b: str):
    """Compare two historical experiments side-by-side."""
    dir_a = Path("experiments") / exp_a
    dir_b = Path("experiments") / exp_b

    if not dir_a.exists() or not dir_b.exists():
        raise FileNotFoundError(f"One or both experiment directories do not exist: {exp_a}, {exp_b}")

    stats_a_file = dir_a / "failure_analysis.json"
    stats_b_file = dir_b / "failure_analysis.json"
    meta_a_file = dir_a / "metadata.json"
    meta_b_file = dir_b / "metadata.json"

    if not stats_a_file.exists() or not stats_b_file.exists():
        raise FileNotFoundError("Missing failure_analysis.json files in target directories.")

    with open(stats_a_file) as f:
        sa = json.load(f)
    with open(stats_b_file) as f:
        sb = json.load(f)

    with open(meta_a_file) as f:
        ma = json.load(f)
    with open(meta_b_file) as f:
        mb = json.load(f)

    print(f"\n=== Experiment Comparison: {exp_a} vs {exp_b} ===")
    print(f"Git Commit:  {ma.get('git_commit', 'N/A')}  vs  {mb.get('git_commit', 'N/A')}")
    print(f"GPU Model:   {ma.get('gpu_model', 'N/A')}  vs  {mb.get('gpu_model', 'N/A')}")
    
    total_fails_a = sa.get("failed_responses", 0)
    total_fails_b = sb.get("failed_responses", 0)
    delta_fails = total_fails_b - total_fails_a

    print(f"Failed Responses: {total_fails_a} -> {total_fails_b} (Delta: {delta_fails:+d})")

    # Generate regression warnings
    if delta_fails > 0:
        print("\n[WARNING] Regression Warning: The failure count has increased in the second experiment!")
    else:
        print("\n[SUCCESS] Progress Checked: The failure count decreased or remained steady.")

    print("\nDelta breakdown completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Orchestrate training and benchmarking experiments.")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to config file")
    parser.add_argument("--resume", default=None, help="Experiment ID to resume from")
    parser.add_argument("--compare", nargs=2, metavar=("EXP_A", "EXP_B"), help="Compare two experiments")
    args = parser.parse_args()

    # 1. Comparison Mode
    if args.compare:
        run_comparison(args.compare[0], args.compare[1])
        return

    # 2. Setup ID & Dir
    if args.resume:
        exp_id = args.resume
        exp_dir = Path("experiments") / exp_id
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment folder not found to resume: {exp_dir}")
        print(f"Resuming experiment: {exp_id}")
    else:
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_dir = Path("experiments") / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Commencing new experiment: {exp_id}")

    # Load Configuration
    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 3. Safety validations
    train_path = config["data"]["train_path"]
    checksum = get_dataset_checksum(train_path)
    if checksum == "missing":
        raise FileNotFoundError(f"Safety Check Failure: Training dataset missing at {train_path}")

    check_disk_space(config["training"]["output_dir"], config["data"].get("min_free_disk_gb", 5.0))
    print(f"Safety checks passed. Dataset SHA256 Checksum: {checksum}")

    # Save reproducibility metadata
    meta = gather_metadata(config["training"].get("seed", 42))
    meta["dataset_checksum"] = checksum
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Override checkpoint path for this specific experiment directory
    exp_adapter_dir = exp_dir / "adapter"
    config["training"]["output_dir"] = str(exp_dir / "checkpoints")
    config["training"]["final_adapter_dir"] = str(exp_adapter_dir)
    
    # Save active config
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # 4. Start QLoRA training
    print("\n--- [Stage 1/4] Starting QLoRA Training ---")
    run_training_step(str(exp_dir / "config.yaml"), str(exp_dir / "checkpoints"))

    # 5. Benchmark the trained adapter
    print("\n--- [Stage 2/4] Benchmarking Trained Adapter ---")
    run_benchmarking_step(str(exp_adapter_dir))

    # Copy output adapter results to experiment archive
    shutil.copy("logs/adapter_results.json", exp_dir / "benchmark.json")

    # 6. Run failure analysis
    print("\n--- [Stage 3/4] Running Failure Analysis ---")
    run_failure_analysis_step()

    # Copy failure reports
    shutil.copy("logs/failure_statistics.json", exp_dir / "failure_analysis.json")
    if os.path.exists("logs/plots"):
        shutil.copytree("logs/plots", exp_dir / "plots", dirs_exist_ok=True)

    # 7. Generate Experiment Report
    print("\n--- [Stage 4/4] Compiling Final Experiment Report ---")
    
    with open(exp_dir / "failure_analysis.json") as f:
        fails_stat = json.load(f)

    report_content = (
        f"# Experiment Report: {exp_id}\n\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Base Model: {config['model']['name']}\n"
        f"Dataset Checksum: {checksum}\n\n"
        f"## Training Parameters\n"
        f"- **Learning Rate:** {config['training']['learning_rate']}\n"
        f"- **Epochs:** {config['training']['num_train_epochs']}\n"
        f"- **LoRA r:** {config['lora']['r']}\n"
        f"- **LoRA alpha:** {config['lora']['alpha']}\n\n"
        f"## Benchmark & Failure Analysis Summary\n"
        f"- **Successful Responses:** {fails_stat.get('successful_responses', 0)}\n"
        f"- **Failed Responses:** {fails_stat.get('failed_responses', 0)}\n"
        f"- **Completion Rate:** {(fails_stat.get('successful_responses', 0) / fails_stat.get('total_samples', 1) * 100):.1f}%\n\n"
        f"Detailed reports, metrics, plots, and adapters have been safely archived in the experiment folder.\n"
    )

    with open(exp_dir / "report.md", "w") as f:
        f.write(report_content)

    print(f"\nExperiment lifecycle completed successfully. All outputs saved to: {exp_dir}")


if __name__ == "__main__":
    main()
