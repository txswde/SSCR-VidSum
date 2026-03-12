"""
TVSum 统一消融实验评估 & 可视化脚本  v2.0
=========================================

核心修正 (vs v1):
  - gtscore 在 h5 中是 segment-level (len==len(picks)==features.shape[0])
  - 直接在 segment-level 计算 Spearman/Kendall，不做逐帧上采样
  - 严格长度一致性保护 + 常数序列容忍度
  - 同时报告 macro-average 与 weighted-average
  - 评估时分别使用 uniform mask 和 all-ones mask (sanity check)

功能:
  Part 1  日志解析 (无需 GPU)
  Part 2  Rank Correlation 评估 (需要 GPU / CPU)
  Part 3  可视化
  Part 4  Markdown 报告生成

Usage:
    python evaluate_tvsum_ablation.py                           # Full evaluation
    python evaluate_tvsum_ablation.py --skip-eval               # Log parsing + viz only
    python evaluate_tvsum_ablation.py --skip-eval --report-only # Report only
"""

import os
import sys
import re
import json
import argparse
import numpy as np
from collections import OrderedDict

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, 'logs')

# 2×2 Grid definition: (display_name -> (dir_name, DCR, PSAL))
TVSUM_GRID = OrderedDict([
    ('Baseline',   ('TVSum_tvsum_baseline',       False, False)),
    ('DCR Only',   ('TVSum_causal_contrastive',   True,  False)),
    ('PSAL Only',  ('TVSum_tvsum_psal_only',      False, True)),
    ('Full',       ('TVSum_full_causal',           True,  True)),
])

COLORS = {
    'Baseline':  '#95a5a6',
    'DCR Only':  '#3498db',
    'PSAL Only': '#2ecc71',
    'Full':      '#e74c3c',
}


# ===========================================================================
# Part 1: Log Parsing
# ===========================================================================

def parse_tvsum_log(log_path):
    """Parse a TVSum training log file to extract per-epoch F1 and final results."""
    result = {
        'eval_epochs': [],
        'final_f1': None,
        'split_f1': {},
    }
    
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    eval_pattern = re.compile(r'\[Eval\]\s+Epoch:\s+(\d+)/(\d+)\s+F-score:\s+([\d.]+)/([\d.]+)')
    f1_pattern = re.compile(r'F1-score:\s+([\d.]+)')
    f1_results_pattern = re.compile(r"F1_results:\s+(\{.+\})")
    
    for line in lines:
        m = eval_pattern.search(line)
        if m:
            result['eval_epochs'].append({
                'epoch': int(m.group(1)),
                'max_epoch': int(m.group(2)),
                'f1_current': float(m.group(3)),
                'f1_best': float(m.group(4)),
            })
        
        m = f1_pattern.search(line)
        if m:
            result['final_f1'] = float(m.group(1))
        
        m = f1_results_pattern.search(line)
        if m:
            try:
                result['split_f1'] = eval(m.group(1))
            except:
                pass
    
    return result


def collect_all_logs():
    """Collect parsed logs for all models in the grid."""
    data = {}
    for name, (dir_name, dcr, psal) in TVSUM_GRID.items():
        log_path = os.path.join(LOG_DIR, dir_name, 'log.txt')
        parsed = parse_tvsum_log(log_path)
        if parsed:
            data[name] = parsed
            f1_str = f"{parsed['final_f1']:.4f}" if parsed['final_f1'] else "N/A"
            print(f"  ✓ {name:15s} ({dir_name}): F1 = {f1_str}")
        else:
            print(f"  ✗ {name:15s} ({dir_name}): log not found")
    return data


# ===========================================================================
# Part 2: Rank Correlation — SEGMENT-LEVEL (v2)
# ===========================================================================

def evaluate_rank_correlation_all():
    """
    Segment-level Rank Correlation.
    
    Key design decisions:
      (a) gtscore is at segment-level (len==features.shape[0]==len(picks))
      (b) pred_scores from model is also segment-level
      (c) Compare directly, no upsampling → avoids ties from duplicated values
      (d) Report macro-average and weighted-average (by #segments)
      (e) Test with both uniform-bucket mask AND all-ones mask
    """
    import torch
    import h5py
    import yaml
    import math
    from scipy.stats import spearmanr, kendalltau
    from models import Model_VideoSumm
    from config import build_args
    
    # Build base args for TVSum
    sys.argv = [sys.argv[0], '--dataset', 'TVSum']
    base_args = build_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    
    h5_path = os.path.join('data', 'TVSum', 'feature', 'eccv16_dataset_tvsum_google_pool5.h5')
    text_path = os.path.join('data', 'TVSum', 'feature', 'text_roberta.npy')
    split_path = os.path.join('data', 'TVSum', 'splits.yml')
    
    for p in [h5_path, text_path, split_path]:
        if not os.path.exists(p):
            print(f"  ❌ Missing: {p}")
            return {}
    
    video_h5 = h5py.File(h5_path, 'r')
    text_feature_dict = np.load(text_path, allow_pickle=True).item()
    with open(split_path, 'r') as f:
        splits = yaml.safe_load(f)
    
    # We test with two mask modes: 'uniform' (linear bucket) and 'allones' (full attention)
    MASK_MODES = ['uniform', 'allones']
    
    results = {}
    
    for model_name, (dir_name, dcr, psal) in TVSUM_GRID.items():
        checkpoint_dir = os.path.join(LOG_DIR, dir_name, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            print(f"  ✗ {model_name}: checkpoint dir not found")
            continue
        
        print(f"\n  ━━ Evaluating: {model_name} ━━")
        
        model_results = {}
        
        for mask_mode in MASK_MODES:
            per_video_rho = []
            per_video_tau = []
            per_video_nsegs = []  # for weighted average
            n_skipped_const = 0
            n_skipped_mismatch = 0
            n_total = 0
            
            for split_idx in range(len(splits)):
                ckpt_path = os.path.join(checkpoint_dir, f'model_best_split{split_idx}.pt')
                if not os.path.exists(ckpt_path):
                    continue
                
                model = Model_VideoSumm(args=base_args)
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(device)
                model.eval()
                
                test_keys = splits[split_idx]['test_keys']
                
                with torch.no_grad():
                    for key in test_keys:
                        video_name = key.split('/')[-1]
                        if video_name not in video_h5:
                            continue
                        
                        n_total += 1
                        video_file = video_h5[video_name]
                        
                        # ---- Load at segment level ----
                        features = video_file['features'][...].astype(np.float32)  # [T_seg, 1024]
                        gtscore = video_file['gtscore'][...].astype(np.float32)    # [T_seg] or [n_frames]
                        picks = video_file['picks'][...].astype(np.int32)          # [T_seg]
                        n_frames = int(video_file['n_frames'][...])
                        
                        T_seg = features.shape[0]
                        
                        # ---- Length-branch: determine gtscore granularity ----
                        if len(gtscore) == T_seg:
                            # gtscore is at segment/pick level — use directly
                            gt_seg = gtscore.copy()
                        elif len(gtscore) == n_frames:
                            # gtscore is at full-frame level — aggregate to segment level
                            gt_seg = np.zeros(T_seg, dtype=np.float32)
                            for i in range(T_seg):
                                lo = picks[i]
                                hi = picks[i + 1] if i + 1 < T_seg else n_frames
                                gt_seg[i] = gtscore[lo:hi].mean()
                        else:
                            # Unknown format — skip with log
                            n_skipped_mismatch += 1
                            continue
                        
                        # Normalize GT to [0,1]
                        gt_range = gt_seg.max() - gt_seg.min()
                        if gt_range > 1e-8:
                            gt_seg = (gt_seg - gt_seg.min()) / gt_range
                        
                        # ---- Forward pass ----
                        video = torch.from_numpy(features).unsqueeze(0).to(device)
                        text = torch.from_numpy(text_feature_dict[video_name]).to(torch.float32).unsqueeze(0).to(device)
                        
                        num_frame = video.shape[1]
                        num_sentence = text.shape[1]
                        mask_video = torch.ones(1, num_frame, dtype=torch.long).to(device)
                        mask_text = torch.ones(1, num_sentence, dtype=torch.long).to(device)
                        
                        video_cls_label = torch.zeros(1, num_frame).to(device)
                        text_label = torch.zeros(1, num_sentence).to(device)
                        
                        # ---- Construct mask based on mode ----
                        if mask_mode == 'allones':
                            video_to_text_mask = torch.ones((num_frame, num_sentence), dtype=torch.long)
                            text_to_video_mask = torch.ones((num_sentence, num_frame), dtype=torch.long)
                        else:  # uniform
                            frame_sentence_ratio = int(math.ceil(num_frame / num_sentence))
                            video_to_text_mask = torch.zeros((num_frame, num_sentence), dtype=torch.long)
                            text_to_video_mask = torch.zeros((num_sentence, num_frame), dtype=torch.long)
                            for j in range(num_sentence):
                                s = j * frame_sentence_ratio
                                e = min((j + 1) * frame_sentence_ratio, num_frame)
                                video_to_text_mask[s:e, j] = 1
                                text_to_video_mask[j, s:e] = 1
                        
                        video_to_text_mask_list = [video_to_text_mask.to(device)]
                        text_to_video_mask_list = [text_to_video_mask.to(device)]
                        
                        outputs = model(
                            video=video, text=text,
                            mask_video=mask_video, mask_text=mask_text,
                            video_label=video_cls_label, text_label=text_label,
                            video_to_text_mask_list=video_to_text_mask_list,
                            text_to_video_mask_list=text_to_video_mask_list
                        )
                        
                        pred_scores = outputs['video_pred_cls'].sigmoid().cpu().numpy()[0]  # [T_seg]
                        
                        # ---- Length consistency protection ----
                        L = min(len(pred_scores), len(gt_seg))
                        if L != len(pred_scores) or L != len(gt_seg):
                            n_skipped_mismatch += 1
                            # Still proceed with min length
                        
                        pred_s = pred_scores[:L]
                        gt_s = gt_seg[:L]
                        
                        # ---- Robust constant-sequence check ----
                        if np.std(pred_s) < 1e-8 or np.std(gt_s) < 1e-8:
                            n_skipped_const += 1
                            continue
                        
                        # ---- Compute rank correlations ----
                        rho, _ = spearmanr(pred_s, gt_s, nan_policy='omit')
                        tau, _ = kendalltau(pred_s, gt_s, nan_policy='omit')
                        
                        if not np.isnan(rho) and not np.isnan(tau):
                            per_video_rho.append(rho)
                            per_video_tau.append(tau)
                            per_video_nsegs.append(L)
                
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # ---- Aggregation ----
            n_valid = len(per_video_rho)
            if n_valid > 0:
                # Macro (equal weight per video)
                macro_rho = np.mean(per_video_rho)
                macro_tau = np.mean(per_video_tau)
                
                # Weighted (by number of segments)
                weights = np.array(per_video_nsegs, dtype=np.float64)
                weights /= weights.sum()
                weighted_rho = np.average(per_video_rho, weights=weights)
                weighted_tau = np.average(per_video_tau, weights=weights)
            else:
                macro_rho = macro_tau = weighted_rho = weighted_tau = 0.0
            
            model_results[mask_mode] = {
                'macro_spearman': float(macro_rho),
                'macro_kendall': float(macro_tau),
                'weighted_spearman': float(weighted_rho),
                'weighted_kendall': float(weighted_tau),
                'n_valid': n_valid,
                'n_total': n_total,
                'n_skipped_const': n_skipped_const,
                'n_skipped_mismatch': n_skipped_mismatch,
            }
            
            print(f"    [{mask_mode:8s}] ρ_macro={macro_rho:.4f}  τ_macro={macro_tau:.4f}  "
                  f"ρ_wt={weighted_rho:.4f}  τ_wt={weighted_tau:.4f}  "
                  f"({n_valid}/{n_total} valid, {n_skipped_const} const, {n_skipped_mismatch} mismatch)")
        
        results[model_name] = model_results
    
    video_h5.close()
    return results


# ===========================================================================
# Part 3: Visualization
# ===========================================================================

def create_ablation_bar_chart(log_data, rank_data, save_dir):
    """Create 2×2 ablation bar chart: F1, Spearman(macro), Kendall(macro)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    models = list(TVSUM_GRID.keys())
    
    f1_vals = [log_data.get(m, {}).get('final_f1', 0) or 0 for m in models]
    
    # Use uniform mask results as primary
    rho_vals = [rank_data.get(m, {}).get('uniform', {}).get('macro_spearman', 0) for m in models]
    tau_vals = [rank_data.get(m, {}).get('uniform', {}).get('macro_kendall', 0) for m in models]
    
    if all(v == 0 for v in rho_vals):
        # Fall back to any available data
        rho_vals = [rank_data.get(m, {}).get('allones', {}).get('macro_spearman', 0) for m in models]
        tau_vals = [rank_data.get(m, {}).get('allones', {}).get('macro_kendall', 0) for m in models]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics_data = [
        ('F1-Score (5-fold avg)', f1_vals),
        ("Spearman's ρ (macro)", rho_vals),
        ("Kendall's τ (macro)", tau_vals),
    ]
    
    for ax, (metric_name, vals) in zip(axes, metrics_data):
        colors = [COLORS.get(m, '#333') for m in models]
        bars = ax.bar(models, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
        
        for bar, val in zip(bars, vals):
            if val != 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title(metric_name, fontsize=13, fontweight='bold')
        
        # Dynamic y-axis
        valid = [v for v in vals if v != 0]
        if valid:
            lo = min(valid) * 0.9
            hi = max(valid) * 1.1
            ax.set_ylim(lo, hi)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', rotation=15)
    
    plt.suptitle('TVSum Ablation: DCR × PSAL Generalization', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    path = os.path.join(save_dir, 'tvsum_ablation_bar_chart.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Bar chart: {path}")


def create_mask_sanity_check_chart(rank_data, save_dir):
    """Sanity check: uniform vs allones mask comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    models = [m for m in TVSUM_GRID if m in rank_data and 'uniform' in rank_data[m] and 'allones' in rank_data[m]]
    if not models:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(models))
    width = 0.35
    
    for ax, metric, label in zip(axes,
                                  ['macro_spearman', 'macro_kendall'],
                                  ["Spearman's ρ", "Kendall's τ"]):
        uniform_vals = [rank_data[m]['uniform'][metric] for m in models]
        allones_vals = [rank_data[m]['allones'][metric] for m in models]
        
        bars1 = ax.bar(x - width/2, uniform_vals, width, label='Uniform mask', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, allones_vals, width, label='All-ones mask', color='#e67e22', alpha=0.8)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.002,
                       f'{h:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.set_title(f'{label}: Mask Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('Sanity Check: Uniform vs All-Ones Mask', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(save_dir, 'tvsum_mask_sanity_check.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Mask sanity check: {path}")


def create_training_curves(log_data, save_dir):
    """Plot F1 training curves over epochs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, data in log_data.items():
        if not data or not data['eval_epochs']:
            continue
        epochs = [e['epoch'] for e in data['eval_epochs']]
        f1_best = [e['f1_best'] for e in data['eval_epochs']]
        f1_current = [e['f1_current'] for e in data['eval_epochs']]
        
        color = COLORS.get(name, '#333')
        ax.plot(epochs, f1_current, color=color, alpha=0.3, linewidth=0.8)
        ax.plot(epochs, f1_best, color=color, linewidth=2, label=f'{name} (best: {max(f1_best):.4f})')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('TVSum Training Curves: Best F1 per Epoch', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'tvsum_training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Training curves: {path}")


def create_delta_heatmap(log_data, rank_data, save_dir):
    """Δ improvement heatmap over baseline (F1 + rank-corr)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if 'Baseline' not in log_data or not log_data['Baseline']['final_f1']:
        print("  ⚠ Baseline not available for delta heatmap.")
        return
    
    baseline_f1 = log_data['Baseline']['final_f1']
    baseline_rho = rank_data.get('Baseline', {}).get('uniform', {}).get('macro_spearman', 0)
    baseline_tau = rank_data.get('Baseline', {}).get('uniform', {}).get('macro_kendall', 0)
    
    models = ['DCR Only', 'PSAL Only', 'Full']
    metrics = ['F1-Score', "Spearman's ρ", "Kendall's τ"]
    
    delta = np.zeros((len(models), len(metrics)))
    
    for i, m in enumerate(models):
        f1 = log_data.get(m, {}).get('final_f1', 0) or 0
        rho = rank_data.get(m, {}).get('uniform', {}).get('macro_spearman', 0)
        tau = rank_data.get(m, {}).get('uniform', {}).get('macro_kendall', 0)
        
        delta[i, 0] = (f1 - baseline_f1) / baseline_f1 * 100 if baseline_f1 else 0
        delta[i, 1] = (rho - baseline_rho) / abs(baseline_rho) * 100 if abs(baseline_rho) > 1e-8 else 0
        delta[i, 2] = (tau - baseline_tau) / abs(baseline_tau) * 100 if abs(baseline_tau) > 1e-8 else 0
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    vmax = max(abs(delta.max()), abs(delta.min()), 5)
    im = ax.imshow(delta, cmap=plt.cm.RdYlGn, aspect='auto', vmin=-vmax, vmax=vmax)
    
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)
    
    for i in range(len(models)):
        for j in range(len(metrics)):
            val = delta[i, j]
            sign = '+' if val > 0 else ''
            ax.text(j, i, f'{sign}{val:.2f}%', ha='center', va='center',
                   fontsize=12, fontweight='bold',
                   color='white' if abs(val) > vmax*0.6 else 'black')
    
    ax.set_title('Δ Improvement over Baseline (%)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='% Change')
    plt.tight_layout()
    
    path = os.path.join(save_dir, 'tvsum_delta_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Delta heatmap: {path}")


# ===========================================================================
# Part 4: Report
# ===========================================================================

def generate_markdown_report(log_data, rank_data, save_dir):
    """Generate comprehensive markdown report."""
    from datetime import datetime
    
    L = []
    L.append("# TVSum 创新点泛化评估报告\n")
    L.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    L.append("## 评估方法说明\n")
    L.append("- **Rank Correlation** 在 **segment-level** 计算（不做逐帧上采样），避免 ties 造成的失真")
    L.append("- 同时报告 **macro-average**（每个视频等权）和 **weighted-average**（按 segment 数量加权）")
    L.append("- 对两种 attention mask 进行 sanity check：uniform（线性分桶）和 all-ones（全连接）\n")
    
    # ---- F1 2×2 表 ----
    L.append("## 2×2 消融结果\n")
    L.append("### F1-Score (5-Fold Cross Validation)\n")
    L.append("| | PSAL OFF | PSAL ON |")
    L.append("|---|---|---|")
    
    def fmt_f1(name):
        v = log_data.get(name, {}).get('final_f1') if name in log_data else None
        return f"**{v:.4f}**" if v else "N/A"
    
    L.append(f"| **DCR OFF** | {fmt_f1('Baseline')} | {fmt_f1('PSAL Only')} |")
    L.append(f"| **DCR ON** | {fmt_f1('DCR Only')} | {fmt_f1('Full')} |")
    L.append("")
    
    # ---- Rank Correlation 表 ----
    if rank_data:
        for mask_mode in ['uniform', 'allones']:
            mask_label = 'Uniform Mask' if mask_mode == 'uniform' else 'All-Ones Mask'
            L.append(f"### Rank Correlation — {mask_label}\n")
            L.append("| Model | ρ (macro) | τ (macro) | ρ (weighted) | τ (weighted) | Valid/Total |")
            L.append("|---|---|---|---|---|---|")
            for name in TVSUM_GRID:
                if name in rank_data and mask_mode in rank_data[name]:
                    d = rank_data[name][mask_mode]
                    L.append(f"| {name} | {d['macro_spearman']:.4f} | {d['macro_kendall']:.4f} | "
                            f"{d['weighted_spearman']:.4f} | {d['weighted_kendall']:.4f} | "
                            f"{d['n_valid']}/{d['n_total']} |")
            L.append("")
    
    # ---- Per-split F1 ----
    L.append("## 逐 Split 详情\n")
    for name in TVSUM_GRID:
        if name in log_data and log_data[name].get('split_f1'):
            L.append(f"### {name}\n")
            L.append("| Split | F1-Score |")
            L.append("|---|---|")
            for split, f1 in sorted(log_data[name]['split_f1'].items()):
                L.append(f"| {split} | {f1:.4f} |")
            L.append("")
    
    # ---- Visualizations ----
    L.append("## 可视化\n")
    for fname, title in [
        ('tvsum_ablation_bar_chart.png', '消融柱状图'),
        ('tvsum_training_curves.png', '训练曲线'),
        ('tvsum_delta_heatmap.png', 'Δ 改进热力图'),
        ('tvsum_mask_sanity_check.png', 'Mask Sanity Check'),
    ]:
        path = os.path.join(save_dir, fname)
        if os.path.exists(path):
            L.append(f"### {title}\n")
            L.append(f"![{title}]({path})\n")
    
    # ---- Analysis ----
    L.append("## 分析\n")
    if 'Baseline' in log_data and log_data['Baseline'].get('final_f1'):
        baseline_f1 = log_data['Baseline']['final_f1']
        for name in ['DCR Only', 'PSAL Only', 'Full']:
            if name in log_data and log_data[name].get('final_f1'):
                f1 = log_data[name]['final_f1']
                delta = (f1 - baseline_f1) / baseline_f1 * 100
                symbol = '↑' if delta > 0 else '↓'
                L.append(f"- **{name}** vs Baseline: F1 {symbol} {abs(delta):.2f}% ({baseline_f1:.4f} → {f1:.4f})")
        L.append("")
    
    report_path = os.path.join(save_dir, 'tvsum_ablation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    
    print(f"  ✓ Report: {report_path}")
    return report_path


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='TVSum Unified Ablation Evaluation v2')
    parser.add_argument('--skip-eval', action='store_true', help='Skip rank correlation evaluation (GPU)')
    parser.add_argument('--report-only', action='store_true', help='Only regenerate report from cached data')
    parser.add_argument('--save-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    save_dir = args.save_dir or os.path.join(LOG_DIR, 'analysis', 'tvsum_ablation')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  TVSum Unified Ablation Evaluation v2.0")
    print(f"  Features: segment-level corr, macro+weighted avg, mask sanity check")
    print(f"  Output: {save_dir}")
    print(f"{'='*70}")
    
    # --- Part 1: Log Parsing ---
    print(f"\n[Part 1] Parsing training logs...")
    log_data = collect_all_logs()
    
    cache_path = os.path.join(save_dir, 'log_data.json')
    serializable = {}
    for name, data in log_data.items():
        serializable[name] = {
            'final_f1': data['final_f1'],
            'split_f1': data['split_f1'],
            'num_eval_epochs': len(data['eval_epochs']),
        }
    with open(cache_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    # --- Part 2: Rank Correlation ---
    rank_data = {}
    rank_cache = os.path.join(save_dir, 'rank_data.json')
    
    if args.report_only and os.path.exists(rank_cache):
        print(f"\n[Part 2] Loading cached rank correlation data...")
        with open(rank_cache, 'r') as f:
            rank_data = json.load(f)
    elif not args.skip_eval:
        print(f"\n[Part 2] Computing segment-level rank correlation...")
        rank_data = evaluate_rank_correlation_all()
        with open(rank_cache, 'w') as f:
            json.dump(rank_data, f, indent=2)
    else:
        print(f"\n[Part 2] Skipped.")
        if os.path.exists(rank_cache):
            with open(rank_cache, 'r') as f:
                rank_data = json.load(f)
            print(f"  Loaded cached rank data from previous run.")
    
    # --- Part 3: Visualization ---
    print(f"\n[Part 3] Generating visualizations...")
    create_ablation_bar_chart(log_data, rank_data, save_dir)
    create_training_curves(log_data, save_dir)
    create_delta_heatmap(log_data, rank_data, save_dir)
    if rank_data:
        create_mask_sanity_check_chart(rank_data, save_dir)
    
    # --- Part 4: Report ---
    print(f"\n[Part 4] Generating report...")
    report_path = generate_markdown_report(log_data, rank_data, save_dir)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Output: {save_dir}")
    print(f"\n  {'Model':15s} {'F1':>8s} {'ρ(macro)':>10s} {'τ(macro)':>10s} {'ρ(wt)':>10s} {'τ(wt)':>10s}")
    print(f"  {'-'*63}")
    for name in TVSUM_GRID:
        f1 = log_data.get(name, {}).get('final_f1')
        rd = rank_data.get(name, {}).get('uniform', {})
        f1s = f"{f1:.4f}" if f1 else "N/A"
        rho_m = f"{rd.get('macro_spearman', 0):.4f}" if rd else "N/A"
        tau_m = f"{rd.get('macro_kendall', 0):.4f}" if rd else "N/A"
        rho_w = f"{rd.get('weighted_spearman', 0):.4f}" if rd else "N/A"
        tau_w = f"{rd.get('weighted_kendall', 0):.4f}" if rd else "N/A"
        print(f"  {name:15s} {f1s:>8s} {rho_m:>10s} {tau_m:>10s} {rho_w:>10s} {tau_w:>10s}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
