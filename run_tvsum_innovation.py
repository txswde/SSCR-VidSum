"""
TVSum Adaptive Causal Alignment Experiment
==========================================
Compares the traditional Linear Alignment with the proposed Adaptive Causal Alignment.

Usage:
    python run_tvsum_innovation.py [--run_all] [--experiment NAME]

Experiments:
    1. baseline_linear:   Traditional linear alignment assumption.
    2. adaptive_causal:   Innovation: Causal Alignment Discovery replaces the linear heuristic.
"""

import subprocess
import argparse
import os
import time
from datetime import datetime

# Base command for TVSum training
# Reduced batch size to 1 to avoid CUDA OOM (4GB VRAM constraint)
BASE_CMD = "python train.py --dataset TVSum --max_epoch 30 --batch_size 1 --print_freq 50 --eval_freq 2"

# Experiment configurations
EXPERIMENTS = [
    {
        "name": "baseline_linear",
        "description": "Baseline: Traditional linear alignment (frame_i <-> sentence_i/N)",
        "cmd": BASE_CMD,  # Default behavior uses linear alignment mask
        "model_dir": "logs/TVSum_baseline_linear"
    },
    {
        "name": "adaptive_causal",
        "description": "Innovation: Adaptive Causal Alignment (Intervention Dissimilarity)",
        "cmd": f"{BASE_CMD} --disable_alignment_mask --enable_causal_alignment --lambda_causal_alignment 0.5",
        "model_dir": "logs/TVSum_adaptive_causal"
    },
]

def run_experiment(exp_config, dry_run=False):
    """Run a single experiment."""
    name = exp_config["name"]
    desc = exp_config["description"]
    cmd = exp_config["cmd"]
    model_dir = exp_config["model_dir"]
    
    # Append model_dir to command
    full_cmd = f"{cmd} --model_dir {model_dir}"
    
    print("\n" + "=" * 70)
    print(f"Experiment: {name}")
    print(f"Description: {desc}")
    print(f"Command: {full_cmd}")
    print("=" * 70)
    
    if dry_run:
        print("[DRY RUN] Skipping actual execution.")
        return None
    
    # Create log directory
    os.makedirs(model_dir, exist_ok=True)
    
    log_file = os.path.join(model_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    print(f"Logging to: {log_file}")
    
    start_time = time.time()
    
    # Run the command
    with open(log_file, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            full_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end="")
            f.write(line)
        
        process.wait()
    
    elapsed = time.time() - start_time
    
    if process.returncode != 0:
        print(f"\n❌ Experiment '{name}' FAILED with return code {process.returncode}.")
        print(f"  Check log: {log_file}")
    else:
        print(f"\n✓ Experiment '{name}' completed in {elapsed/60:.1f} minutes.")
        print(f"  Model saved to: {model_dir}")
    
    return process.returncode

def main():
    parser = argparse.ArgumentParser(description="TVSum Adaptive Causal Alignment Experiments")
    parser.add_argument("--run_all", action="store_true", help="Run all experiments sequentially")
    parser.add_argument("--experiment", type=str, default=None, 
                        help="Run a specific experiment by name (e.g., 'baseline_linear' or 'adaptive_causal')")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("TVSum Adaptive Causal Alignment Experiment Suite")
    print("=" * 70)
    
    print("\nAvailable experiments:")
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"  {i}. {exp['name']}: {exp['description']}")
    
    experiments_to_run = []
    
    if args.run_all:
        experiments_to_run = EXPERIMENTS
    elif args.experiment:
        for exp in EXPERIMENTS:
            if exp["name"] == args.experiment:
                experiments_to_run.append(exp)
                break
        if not experiments_to_run:
            print(f"\n❌ Experiment '{args.experiment}' not found.")
            return
    else:
        print("\nUsage:")
        print("  python run_tvsum_innovation.py --run_all          # Run all experiments")
        print("  python run_tvsum_innovation.py --experiment NAME  # Run a specific experiment")
        print("  python run_tvsum_innovation.py --dry_run          # Preview commands")
        return
    
    print(f"\nWill run {len(experiments_to_run)} experiment(s).")
    
    for exp in experiments_to_run:
        run_experiment(exp, dry_run=args.dry_run)
    
    print("\n" + "=" * 70)
    print("All experiments completed!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Compare F-scores between baseline and innovation.")
    print("  2. Visualize alignment maps using 'tvsum_causal_analysis.py' (to be created).")

if __name__ == "__main__":
    main()
