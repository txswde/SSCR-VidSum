"""
TVSum 2×2 Ablation Experiment Runner: DCR × PSAL
==================================================

DCR  = Differential Counterfactual Rectification (causal contrastive loss)
PSAL = Prior-free Semantic Alignment Learning  (disable_alignment_mask + causal alignment)

Experiment grid:
  1. baseline       : DCR OFF + PSAL OFF  (needs training)
  2. dcr_only       : DCR ON  + PSAL OFF  (existing: TVSum_causal_contrastive)
  3. psal_only      : DCR OFF + PSAL ON   (needs training)
  4. full (dcr+psal) : DCR ON  + PSAL ON  (existing: TVSum_full_causal)

Usage:
    python run_tvsum_ablation.py --dry-run           # Preview commands
    python run_tvsum_ablation.py --part all           # Run missing experiments
    python run_tvsum_ablation.py --part baseline      # Run only baseline
    python run_tvsum_ablation.py --part psal_only     # Run only PSAL-only
"""

import subprocess
import sys
import os
import time

PYTHON = sys.executable
# Match existing training config: batch_size 4, eval_freq 1
# Max epoch 50 to match TVSum_causal_contrastive (the DCR-only model)
BASE_CMD = f"{PYTHON} train.py --dataset TVSum --max_epoch 50 --batch_size 4 --print_freq 5 --eval_freq 1"

EXPERIMENTS = [
    {
        "name": "tvsum_baseline",
        "desc": "DCR OFF + PSAL OFF — pure baseline, no innovations",
        "cmd": f"{BASE_CMD} --suffix tvsum_baseline "
               f"--lambda_causal_contrast 0.0 "
               f"--lambda_causal_effect 0.0",
        # disable_alignment_mask defaults to False (uses linear mask)
        # enable_causal_alignment defaults to False
        "existing_dir": None,  # Needs training
    },
    {
        "name": "causal_contrastive",
        "desc": "DCR ON + PSAL OFF — causal contrastive loss only",
        "cmd": f"{BASE_CMD} --suffix causal_contrastive "
               f"--lambda_causal_contrast 0.5 "
               f"--lambda_causal_effect 0.3",
        "existing_dir": "logs/TVSum_causal_contrastive",  # Already trained
    },
    {
        "name": "tvsum_psal_only",
        "desc": "DCR OFF + PSAL ON — free alignment only, no causal loss",
        "cmd": f"{BASE_CMD} --suffix tvsum_psal_only "
               f"--disable_alignment_mask "
               f"--enable_causal_alignment "
               f"--lambda_causal_contrast 0.0 "
               f"--lambda_causal_effect 0.0 "
               f"--lambda_causal_alignment 0.5",
        "existing_dir": None,  # Needs training
    },
    {
        "name": "full_causal",
        "desc": "DCR ON + PSAL ON — full model with both innovations",
        "cmd": f"{BASE_CMD} --max_epoch 30 --suffix full_causal "
               f"--disable_alignment_mask "  # free alignment  (PSAL)
               f"--enable_causal_alignment "
               f"--lambda_causal_contrast 0.3 "
               f"--lambda_causal_effect 0.3 "
               f"--lambda_causal_alignment 0.5",
        "existing_dir": "logs/TVSum_full_causal",  # Already trained
    },
]


def run_experiment(exp, dry_run=False, skip_existing=True):
    """Run a single experiment."""
    name = exp["name"]
    expected_dir = exp.get("existing_dir") or f"logs/TVSum_{name}"
    
    print(f"\n{'='*70}")
    print(f"  Experiment: {name}")
    print(f"  Description: {exp['desc']}")
    print(f"  Model dir: {expected_dir}")
    print(f"  Command: {exp['cmd']}")
    print(f"{'='*70}")

    # Check if already exists
    if skip_existing and exp.get("existing_dir") and os.path.exists(exp["existing_dir"]):
        log_path = os.path.join(exp["existing_dir"], "log.txt")
        if os.path.exists(log_path):
            print(f"  [SKIP] Already trained: {exp['existing_dir']}")
            # Try to extract final F1 from log
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                for line in reversed(lines):
                    if 'F1-score' in line:
                        print(f"  Result: {line.strip()}")
                        break
            except:
                pass
            return "SKIPPED"

    if dry_run:
        print("  [DRY RUN] Skipping execution.")
        return "DRY_RUN"

    start = time.time()
    result = subprocess.run(exp["cmd"], shell=True, cwd=os.path.dirname(__file__) or ".")
    elapsed = time.time() - start

    success = result.returncode == 0
    status = "SUCCESS" if success else f"FAILED (code {result.returncode})"
    print(f"\n  => {name}: {status} ({elapsed/60:.1f} min)")
    return "OK" if success else "FAIL"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TVSum 2×2 DCR×PSAL Ablation Experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--part", choices=["baseline", "psal_only", "missing", "all"], default="missing",
                        help="baseline=only baseline, psal_only=only psal, missing=both missing, all=all 4")
    parser.add_argument("--force", action="store_true", help="Force re-run even if model exists")
    args = parser.parse_args()

    experiments = []
    if args.part == "baseline":
        experiments = [e for e in EXPERIMENTS if e["name"] == "tvsum_baseline"]
    elif args.part == "psal_only":
        experiments = [e for e in EXPERIMENTS if e["name"] == "tvsum_psal_only"]
    elif args.part == "missing":
        experiments = [e for e in EXPERIMENTS if e.get("existing_dir") is None]
    elif args.part == "all":
        experiments = EXPERIMENTS

    print(f"\n{'#'*70}")
    print(f"  TVSum DCR × PSAL 2×2 Ablation Experiment Runner")
    print(f"  Total experiments: {len(experiments)}")
    print(f"{'#'*70}")

    results = {}
    for exp in experiments:
        result = run_experiment(exp, dry_run=args.dry_run, skip_existing=not args.force)
        results[exp["name"]] = result

    # Summary
    print(f"\n\n{'#'*70}")
    print(f"  SUMMARY")
    print(f"{'#'*70}")
    for name, status in results.items():
        print(f"  {name:30s} {status}")

    # Print the full 2x2 table
    print(f"\n  Complete 2×2 Grid:")
    print(f"  {'':20s} {'PSAL OFF':25s} {'PSAL ON':25s}")
    baseline_status = results.get("tvsum_baseline", "existing" if os.path.exists("logs/TVSum_tvsum_baseline") else "?")
    dcr_status = results.get("causal_contrastive", "existing" if os.path.exists("logs/TVSum_causal_contrastive") else "?")
    psal_status = results.get("tvsum_psal_only", "existing" if os.path.exists("logs/TVSum_tvsum_psal_only") else "?")
    full_status = results.get("full_causal", "existing" if os.path.exists("logs/TVSum_full_causal") else "?")
    print(f"  {'DCR OFF':20s} {str(baseline_status):25s} {str(psal_status):25s}")
    print(f"  {'DCR ON':20s} {str(dcr_status):25s} {str(full_status):25s}")


if __name__ == "__main__":
    main()
