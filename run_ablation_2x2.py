"""
2×2 Ablation Experiment Runner: DCR × PSAL
===========================================

DCR  = Differential Counterfactual Rectification (lambda_causal_contrast > 0)
PSAL = Prior-free Semantic Alignment Learning  (disable_alignment_mask = True)

Experiment grid:
  1. baseline       : DCR OFF + PSAL OFF  (已有: BLiSS_pure_baseline)
  2. dcr_only       : DCR ON  + PSAL OFF  (需新训练)
  3. psal_only      : DCR OFF + PSAL ON   (已有: BLiSS_unsupervised_align_only, 但含对齐发现模块)
  4. full (dcr+psal) : DCR ON  + PSAL ON  (已有: BLiSS_unsupervised)

This script runs:
  Part A: the missing 100% data experiment (dcr_only)
  Part B: all 4 experiments at 20% data ratio
"""

import subprocess
import sys
import os
import time

# Base command
PYTHON = sys.executable
BASE_CMD = f"{PYTHON} train.py --dataset BLiSS --print_freq 50 --eval_freq 5"

# =============================================
# Part A: Missing 100% experiment (dcr_only)
# =============================================
FULL_DATA_EXPERIMENTS = [
    {
        "name": "dcr_only",
        "desc": "DCR ON + PSAL OFF (100% data) — uses GT alignment mask + causal contrastive loss",
        "cmd": f"{BASE_CMD} --max_epoch 20 --batch_size 64 "
               f"--suffix dcr_only "
               f"--lambda_causal_contrast 0.3 "
               f"--lambda_causal_effect 0.2",
        # Note: disable_alignment_mask defaults to False (uses GT mask)
        # Note: enable_causal_alignment defaults to False (no alignment discovery module)
    },
]

# =============================================
# Part B: 20% data ablation (all 4 cells)
# =============================================
RATIO = 0.2

ABLATION_EXPERIMENTS = [
    {
        "name": "ablation_baseline_20pct",
        "desc": f"DCR OFF + PSAL OFF ({int(RATIO*100)}% data)",
        "cmd": f"{BASE_CMD} --max_epoch 20 --batch_size 64 "
               f"--suffix ablation_baseline_20pct "
               f"--data_ratio {RATIO} "
               f"--lambda_causal_contrast 0.0 "
               f"--lambda_causal_effect 0.0",
    },
    {
        "name": "ablation_dcr_only_20pct",
        "desc": f"DCR ON + PSAL OFF ({int(RATIO*100)}% data)",
        "cmd": f"{BASE_CMD} --max_epoch 20 --batch_size 64 "
               f"--suffix ablation_dcr_only_20pct "
               f"--data_ratio {RATIO} "
               f"--lambda_causal_contrast 0.3 "
               f"--lambda_causal_effect 0.2",
    },
    {
        "name": "ablation_psal_only_20pct",
        "desc": f"DCR OFF + PSAL ON ({int(RATIO*100)}% data)",
        "cmd": f"{BASE_CMD} --max_epoch 20 --batch_size 64 "
               f"--suffix ablation_psal_only_20pct "
               f"--data_ratio {RATIO} "
               f"--disable_alignment_mask "
               f"--lambda_causal_contrast 0.0 "
               f"--lambda_causal_effect 0.0",
    },
    {
        "name": "ablation_full_20pct",
        "desc": f"DCR ON + PSAL ON ({int(RATIO*100)}% data)",
        "cmd": f"{BASE_CMD} --max_epoch 20 --batch_size 64 "
               f"--suffix ablation_full_20pct "
               f"--data_ratio {RATIO} "
               f"--disable_alignment_mask "
               f"--lambda_causal_contrast 0.3 "
               f"--lambda_causal_effect 0.2",
    },
]


def run_experiment(exp, dry_run=False):
    """Run a single experiment."""
    print(f"\n{'='*70}")
    print(f"  Experiment: {exp['name']}")
    print(f"  Description: {exp['desc']}")
    print(f"  Command: {exp['cmd']}")
    print(f"{'='*70}\n")

    if dry_run:
        print("[DRY RUN] Skipping execution.")
        return True

    start = time.time()
    result = subprocess.run(exp["cmd"], shell=True, cwd=os.path.dirname(__file__) or ".")
    elapsed = time.time() - start

    success = result.returncode == 0
    status = "SUCCESS" if success else f"FAILED (code {result.returncode})"
    print(f"\n  => {exp['name']}: {status} ({elapsed:.0f}s)")
    return success


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run 2x2 DCR×PSAL ablation experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--part", choices=["a", "b", "all"], default="all",
                        help="a=100%% missing exp, b=20%% ablation, all=both")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip experiments whose log dirs already exist")
    cli_args = parser.parse_args()

    experiments = []
    if cli_args.part in ("a", "all"):
        experiments.extend(FULL_DATA_EXPERIMENTS)
    if cli_args.part in ("b", "all"):
        experiments.extend(ABLATION_EXPERIMENTS)

    print(f"\n{'#'*70}")
    print(f"  DCR × PSAL 2×2 Ablation Experiment Runner")
    print(f"  Total experiments: {len(experiments)}")
    print(f"{'#'*70}")

    results = {}
    for exp in experiments:
        # Check if already exists
        log_dir = f"logs/BLiSS_{exp['name']}"
        if cli_args.skip_existing and os.path.exists(log_dir):
            print(f"\n  [SKIP] {exp['name']}: {log_dir} already exists")
            results[exp["name"]] = "SKIPPED"
            continue

        success = run_experiment(exp, dry_run=cli_args.dry_run)
        results[exp["name"]] = "OK" if success else "FAIL"

    # Summary
    print(f"\n\n{'#'*70}")
    print(f"  SUMMARY")
    print(f"{'#'*70}")
    for name, status in results.items():
        print(f"  {name:40s} {status}")

    # Print the full 2x2 table including existing models
    print(f"\n\n  Complete 2×2 Grid (100% data):")
    print(f"  {'':20s} {'PSAL OFF (GT mask)':25s} {'PSAL ON (free mask)':25s}")
    print(f"  {'DCR OFF':20s} {'pure_baseline ✅':25s} {'unsupervised_align_only* ✅':25s}")
    print(f"  {'DCR ON':20s} {'dcr_only ' + results.get('dcr_only', '?'):25s} {'unsupervised ✅':25s}")
    print(f"\n  * unsupervised_align_only also has enable_causal_alignment=True")

    if cli_args.part in ("b", "all"):
        print(f"\n  2×2 Grid (20% data):")
        print(f"  {'':20s} {'PSAL OFF':25s} {'PSAL ON':25s}")
        r = results
        print(f"  {'DCR OFF':20s} {r.get('ablation_baseline_20pct', '?'):25s} {r.get('ablation_psal_only_20pct', '?'):25s}")
        print(f"  {'DCR ON':20s} {r.get('ablation_dcr_only_20pct', '?'):25s} {r.get('ablation_full_20pct', '?'):25s}")


if __name__ == "__main__":
    main()
