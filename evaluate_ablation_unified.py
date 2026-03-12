"""
BLiSS 统一消融实验评估 & 可视化脚本  v1.0

功能:
  Part 1  日志解析 (无需 GPU)
    - 从每个模型的 log.txt 提取 best/final R1, R2, RL, Cos, Causal
    - 构建 2×2 消融表 (全量 + 20% 数据)

  Part 2  模型级别对齐评估 (需 GPU, 可选)
    - 复用 evaluate_bliss_alignment_improved.py 的指标函数
    - 计算 AUC-PR, F1@fixed, Top-K Hit Rate, Causal Sensitivity

  Part 3  可视化
    - 2×2 消融柱状图
    - 训练曲线 (R1 over epochs)
    - Δ (Delta) 增量贡献表
    - 雷达图 (对齐指标)
    - 综合 Markdown 报告

用法:
    # 仅解析日志 (快速, 无需 GPU):
    python evaluate_ablation_unified.py --skip_alignment

    # 完整评估 (含对齐, 需 GPU):
    python evaluate_ablation_unified.py --num_samples 50
"""

import os
import sys
import re
import json
import argparse
import datetime
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# --- SCI Plotting Standards ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 11
# ------------------------------

# ===========================================================================
# Configuration
# ===========================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLISS_LOG_DIR = os.path.join(SCRIPT_DIR, 'logs', 'BLiss')
BLISS_20PCT_DIR = os.path.join(BLISS_LOG_DIR, 'BLiss_0.2')

# 2×2 Grid definitions:  (display_name, dir_name, DCR, PSAL)
FULL_DATA_GRID = {
    'Baseline':  ('BLiSS_pure_baseline',                     False, False),
    'DCR Only':  ('BLiSS_dcr_only',                          True,  False),
    'PSAL Only': ('BLiSS_unsupervised',                      True,  True),
    'Full':      ('BLiSS_unsupervised_yinguo_duiqi_corrected', True,  True),
}

DATA_20PCT_GRID = {
    'Baseline':  ('BLiSS_ablation_baseline_20pct',  False, False),
    'DCR Only':  ('BLiSS_ablation_dcr_only_20pct',  True,  False),
    'PSAL Only': ('BLiSS_ablation_psal_only_20pct', False, True),
    'Full':      ('BLiSS_ablation_full_20pct',      True,  True),
}

# Supplementary models (not in 2×2 but interesting)
SUPPLEMENTARY = {
    'Align Only': ('BLiSS_unsupervised_align_only', True, False),
}

# Color palette: SCI publication standard
COLORS = {
    'Baseline':  '#7F7F7F',  # Dark Gray
    'DCR Only':  '#D62728',  # Brick Red
    'PSAL Only': '#1F77B4',  # Muted Blue
    'Full':      '#2CA02C',  # Forest Green
    'Align Only': '#FF7F0E', # Safety Orange
}


# ===========================================================================
# Part 1: Log Parsing
# ===========================================================================

def parse_log_file(log_path):
    """Parse a training log file to extract per-epoch metrics.
    
    Returns:
        dict with keys:
            'train_epochs': list of dicts {epoch, R1, R2, RL, Cos}
            'eval_epochs':  list of dicts {epoch, R1, R2, RL, Cos}
            'causal_values': list of floats (running avg per logged iter)
            'best_train': dict of best training metrics
            'best_eval':  dict of best eval metrics
            'config': dict of config from first line
    """
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    train_epochs = []
    eval_epochs = []
    causal_values = []
    config = {}

    # Regex patterns
    # Train epoch summary: [Train] Epoch: X/Y R1: curr/best R2: curr/best RL: curr/best Cos: curr/best
    train_epoch_re = re.compile(
        r'\[Train\]\s+Epoch:\s+(\d+)/(\d+)\s+'
        r'R1:\s+([\d.]+)/([\d.]+)\s+'
        r'R2:\s+([\d.]+)/([\d.]+)\s+'
        r'RL:\s+([\d.]+)/([\d.]+)\s+'
        r'Cos:\s+([\d.]+)/([\d.]+)'
    )
    
    # Eval epoch summary: [Eval]  Epoch: X/Y R1: curr/best R2: curr/best ...
    eval_epoch_re = re.compile(
        r'\[Eval\]\s+Epoch:\s+(\d+)/(\d+)\s+'
        r'R1:\s+([\d.]+)/([\d.]+)\s+'
        r'R2:\s+([\d.]+)/([\d.]+)\s+'
        r'RL:\s+([\d.]+)/([\d.]+)\s+'
        r'Cos:\s+([\d.]+)/([\d.]+)'
    )
    
    # Per-iteration log with Causal
    causal_re = re.compile(r'Causal:\s+([\d.]+)')

    # Config line
    config_re = re.compile(r"\{.*'dataset'.*\}")

    for line in lines:
        # Config
        if not config:
            cfg_match = config_re.search(line)
            if cfg_match:
                try:
                    config = eval(cfg_match.group())
                except:
                    pass

        # Train epoch summary
        m = train_epoch_re.search(line)
        if m:
            train_epochs.append({
                'epoch': int(m.group(1)),
                'max_epoch': int(m.group(2)),
                'R1': float(m.group(3)),
                'R1_best': float(m.group(4)),
                'R2': float(m.group(5)),
                'R2_best': float(m.group(6)),
                'RL': float(m.group(7)),
                'RL_best': float(m.group(8)),
                'Cos': float(m.group(9)),
                'Cos_best': float(m.group(10)),
            })
            continue
        
        # Eval epoch summary (the final line per eval block, with curr/best)
        m = eval_epoch_re.search(line)
        if m:
            eval_epochs.append({
                'epoch': int(m.group(1)),
                'max_epoch': int(m.group(2)),
                'R1': float(m.group(3)),
                'R1_best': float(m.group(4)),
                'R2': float(m.group(5)),
                'R2_best': float(m.group(6)),
                'RL': float(m.group(7)),
                'RL_best': float(m.group(8)),
                'Cos': float(m.group(9)),
                'Cos_best': float(m.group(10)),
            })
            continue

        # Causal values
        cm = causal_re.search(line)
        if cm:
            causal_values.append(float(cm.group(1)))

    # Compute best metrics
    best_train = {}
    if train_epochs:
        last = train_epochs[-1]
        best_train = {
            'R1': last['R1_best'],
            'R2': last['R2_best'],
            'RL': last['RL_best'],
            'Cos': last['Cos_best'],
        }

    best_eval = {}
    if eval_epochs:
        last = eval_epochs[-1]
        best_eval = {
            'R1': last['R1_best'],
            'R2': last['R2_best'],
            'RL': last['RL_best'],
            'Cos': last['Cos_best'],
        }

    return {
        'train_epochs': train_epochs,
        'eval_epochs': eval_epochs,
        'causal_values': causal_values,
        'best_train': best_train,
        'best_eval': best_eval,
        'config': config,
    }


def collect_all_logs(grid_def, base_dir):
    """Collect parsed logs for all models in a grid definition."""
    results = {}
    for cell_name, (dir_name, dcr, psal) in grid_def.items():
        log_path = os.path.join(base_dir, dir_name, 'log.txt')
        parsed = parse_log_file(log_path)
        if parsed is None:
            print(f"  ⚠️  未找到日志: {log_path}")
            continue
        
        # Add metadata
        parsed['cell_name'] = cell_name
        parsed['dir_name'] = dir_name
        parsed['DCR'] = dcr
        parsed['PSAL'] = psal
        parsed['causal_mean'] = float(np.mean(parsed['causal_values'])) if parsed['causal_values'] else 0.0
        parsed['causal_final'] = parsed['causal_values'][-1] if parsed['causal_values'] else 0.0
        
        results[cell_name] = parsed
        
        # Print summary
        be = parsed['best_eval'] or parsed['best_train']
        causal_str = f"Causal={parsed['causal_final']:.4f}" if parsed['causal_values'] else "Causal=N/A"
        print(f"  ✓ {cell_name:12s} | R1={be.get('R1',0):.4f} R2={be.get('R2',0):.4f} "
              f"RL={be.get('RL',0):.4f} Cos={be.get('Cos',0):.4f} | {causal_str}")
    
    return results


# ===========================================================================
# Part 2: Alignment Evaluation (optional, reuses existing code)
# ===========================================================================

def run_alignment_evaluation(grid_def, base_dir, num_samples=50):
    """Run model-based alignment evaluation using existing infrastructure."""
    try:
        import torch
        from config import build_args
        from models import Model_BLiSS
        from datasets import BLiSSDataset
        from evaluate_bliss_alignment_improved import (
            load_model, compute_causal_matrix, compute_improved_metrics, 
            aggregate_metrics, set_global_seed
        )
    except ImportError as e:
        print(f"  ⚠️  无法导入评估模块: {e}")
        return {}

    set_global_seed(42)

    # Build args
    old_argv = sys.argv
    sys.argv = ['', '--dataset=BLiSS']
    args = build_args()
    sys.argv = old_argv
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load test set
    print(f"\n  加载 BLiSS 测试集 (设备={args.device})...")
    test_set = BLiSSDataset(mode='test', args=args)
    print(f"  ✓ 共 {len(test_set)} 个测试样本, 将评估 {num_samples} 个")

    alignment_results = {}

    for cell_name, (dir_name, dcr, psal) in grid_def.items():
        model_path = os.path.join(base_dir, dir_name)
        if not os.path.isdir(model_path):
            print(f"  ⚠️  模型目录不存在: {model_path}")
            continue

        print(f"\n  评估 {cell_name} ({dir_name})...")
        try:
            args.model_dir = model_path
            model = load_model(model_path, args)
        except Exception as e:
            print(f"  ⚠️  模型加载失败: {e}")
            continue

        sample_metrics = []
        n = min(num_samples, len(test_set))

        from tqdm import tqdm
        for idx in tqdm(range(n), desc=f"    分析 {cell_name}"):
            sample = test_set[idx]
            if sample[10].dim() < 2:
                continue
            try:
                causal_np, gt_alignment, sentence_effects = compute_causal_matrix(model, sample, args)
                if gt_alignment.sum() == 0:
                    continue
                metrics = compute_improved_metrics(causal_np, gt_alignment, sentence_effects)
                sample_metrics.append(metrics)
            except Exception as e:
                continue

        if sample_metrics:
            aggregated = aggregate_metrics(sample_metrics)
            aggregated['n_samples'] = len(sample_metrics)
            alignment_results[cell_name] = aggregated
            print(f"    AUC-PR={aggregated['auc_pr_mean']:.4f}  "
                  f"F1@p95={aggregated['f1_at_fixed_mean']:.4f}  "
                  f"Top-K={aggregated['topk_hit_rate_mean']:.4f}")

        # Free GPU
        del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

    return alignment_results


# ===========================================================================
# Part 3: Visualization
# ===========================================================================

def create_ablation_bar_chart(full_data, data_20pct, save_dir):
    """Create 2×2 ablation bar chart for both data regimes."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    cell_order = ['Baseline', 'DCR Only', 'PSAL Only', 'Full']
    metrics = ['R1', 'R2', 'RL', 'Cos']
    x = np.arange(len(metrics))
    width = 0.18

    for ax_idx, (ax, data, title) in enumerate([
        (axes[0], full_data, '(a) Full Data (100%)'),
        (axes[1], data_20pct, '(b) 20% Data'),
    ]):
        for i, cell_name in enumerate(cell_order):
            if cell_name not in data:
                continue
            be = data[cell_name].get('best_eval') or data[cell_name].get('best_train', {})
            vals = [be.get(m, 0) for m in metrics]
            bars = ax.bar(x + i * width, vals, width, 
                         label=cell_name, color=COLORS.get(cell_name, '#999'),
                         alpha=0.85, edgecolor='white', linewidth=0.5)
            # Value labels
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0.3, 0.7)

    plt.suptitle('Ablation Study: DCR and PSAL', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(save_dir, 'ablation_bar_chart.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {path}")
    return path


def create_training_curves(full_data, data_20pct, save_dir):
    """Plot R1 training curves over epochs for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    for ax, data, title in [
        (axes[0], full_data, '(a) Full Data - Training R1'),
        (axes[1], data_20pct, '(b) 20% Data - Training R1'),
    ]:
        for cell_name, parsed in data.items():
            epochs = [e['epoch'] for e in parsed['train_epochs']]
            r1_vals = [e['R1'] for e in parsed['train_epochs']]
            if epochs:
                color = COLORS.get(cell_name, '#999')
                ax.plot(epochs, r1_vals, '-o', color=color, label=cell_name, 
                       linewidth=2, markersize=3, alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('ROUGE-1', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle('Training Dynamics: ROUGE-1 over Epochs', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {path}")
    return path


def create_causal_loss_chart(full_data, data_20pct, save_dir):
    """Plot Causal loss evolution for models that have it."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    
    for ax, data, title in [
        (axes[0], full_data, '(a) Full Data - Causal Loss'),
        (axes[1], data_20pct, '(b) 20% Data - Causal Loss'),
    ]:
        has_data = False
        for cell_name, parsed in data.items():
            cvals = parsed.get('causal_values', [])
            nonzero = [v for v in cvals if v > 0]
            if nonzero:
                color = COLORS.get(cell_name, '#999')
                ax.plot(range(len(nonzero)), nonzero, '-', color=color, 
                       label=f"{cell_name} (avg={np.mean(nonzero):.4f})",
                       linewidth=1.5, alpha=0.8)
                has_data = True
        
        if not has_data:
            ax.text(0.5, 0.5, 'No DCR\n(Causal=0)', ha='center', va='center',
                   fontsize=14, color='#999', transform=ax.transAxes)
        
        ax.set_xlabel('Iteration (logged)', fontsize=11)
        ax.set_ylabel('Causal Loss', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle('Causal Contrastive Loss (DCR) Training Dynamics', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(save_dir, 'causal_loss_curves.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {path}")
    return path


def create_delta_heatmap(full_data, data_20pct, save_dir):
    """Create Δ (delta) contribution heatmap showing improvement over baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    metrics = ['R1', 'R2', 'RL', 'Cos']
    innovations = ['DCR Only', 'PSAL Only', 'Full']

    for ax, data, title in [
        (axes[0], full_data, '(a) Full Data: $\Delta$ vs Baseline'),
        (axes[1], data_20pct, '(b) 20% Data: $\Delta$ vs Baseline'),
    ]:
        if 'Baseline' not in data:
            ax.text(0.5, 0.5, 'Baseline Missing', ha='center', va='center',
                   fontsize=14, transform=ax.transAxes)
            continue

        baseline = data['Baseline'].get('best_eval') or data['Baseline'].get('best_train', {})
        
        delta_matrix = []
        row_labels = []
        for inno_name in innovations:
            if inno_name not in data:
                continue
            be = data[inno_name].get('best_eval') or data[inno_name].get('best_train', {})
            deltas = [(be.get(m, 0) - baseline.get(m, 0)) * 100 for m in metrics]  # percentage points
            delta_matrix.append(deltas)
            row_labels.append(inno_name)
        
        if not delta_matrix:
            continue

        delta_arr = np.array(delta_matrix)
        
        # Custom colormap: SCI standard
        im = ax.imshow(delta_arr, cmap='RdBu', aspect='auto',
                      vmin=-3, vmax=3)
        
        # Labels
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=11)
        
        # Annotate cells
        for i in range(len(row_labels)):
            for j in range(len(metrics)):
                val = delta_arr[i, j]
                sign = '+' if val > 0 else ''
                color = 'white' if abs(val) > 1.5 else 'black'
                ax.text(j, i, f'{sign}{val:.2f}%', ha='center', va='center',
                       fontsize=10, fontweight='bold', color=color)
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Δ (%p)', shrink=0.8)

    plt.suptitle('Incremental Contribution ($\Delta$ = Innovation - Baseline, percentage points)', 
                fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    path = os.path.join(save_dir, 'delta_heatmap.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {path}")
    return path


def create_alignment_radar(alignment_full, alignment_20pct, save_dir):
    """Create radar chart for alignment metrics (if available)."""
    if not alignment_full and not alignment_20pct:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), 
                             subplot_kw=dict(projection='polar'))
    
    radar_keys = ['auc_pr_mean', 'f1_at_fixed_mean', 'topk_hit_rate_mean', 'causal_sensitivity_mean']
    radar_labels = ['AUC-PR', 'F1@p95', 'Top-K Hit', 'Causal Sens.']
    
    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]
    
    for ax, data, title in [
        (axes[0], alignment_full, '(a) Full Data - Alignment Metrics'),
        (axes[1], alignment_20pct, '(b) 20% Data - Alignment Metrics'),
    ]:
        if not data:
            ax.text(0, 0, 'Not Evaluated', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=12, fontweight='bold')
            continue
        
        for cell_name, metrics in data.items():
            values = []
            for k in radar_keys:
                v = metrics.get(k, 0)
                values.append(v if not np.isnan(v) else 0)
            values += values[:1]
            
            color = COLORS.get(cell_name, '#999')
            ax.plot(angles, values, '-o', color=color, linewidth=2, 
                   label=cell_name, markersize=4)
            ax.fill(angles, values, color=color, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.suptitle('Alignment Quality Radar Chart', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.95, 0.93])
    path = os.path.join(save_dir, 'alignment_radar.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {path}")
    return path


# ===========================================================================
# Part 4: Report & Summary
# ===========================================================================

def build_summary_table(data, title):
    """Build a text summary table for a data regime."""
    metrics = ['R1', 'R2', 'RL', 'Cos']
    cell_order = ['Baseline', 'DCR Only', 'PSAL Only', 'Full']
    
    lines = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f" {title}")
    lines.append(f"{'=' * 80}")
    lines.append(f" {'Model':15s} | {'DCR':5s} | {'PSAL':5s} | {'R1':8s} | {'R2':8s} | {'RL':8s} | {'Cos':8s} | {'Causal':8s}")
    lines.append(f" {'-'*15}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    
    baseline_metrics = {}
    for cell_name in cell_order:
        if cell_name not in data:
            lines.append(f" {cell_name:15s} | {'---':>5s} | {'---':>5s} | {'N/A':>8s} | {'N/A':>8s} | {'N/A':>8s} | {'N/A':>8s} | {'N/A':>8s}")
            continue
        
        parsed = data[cell_name]
        be = parsed.get('best_eval') or parsed.get('best_train', {})
        dcr_str = '  ON' if parsed['DCR'] else ' OFF'
        psal_str = '  ON' if parsed['PSAL'] else ' OFF'
        causal_str = f"{parsed.get('causal_final', 0):.4f}"
        
        if cell_name == 'Baseline':
            baseline_metrics = be
        
        lines.append(
            f" {cell_name:15s} | {dcr_str:>5s} | {psal_str:>5s} | "
            f"{be.get('R1', 0):8.4f} | {be.get('R2', 0):8.4f} | "
            f"{be.get('RL', 0):8.4f} | {be.get('Cos', 0):8.4f} | {causal_str:>8s}"
        )
    
    # Delta section
    if baseline_metrics:
        lines.append(f"\n Δ vs Baseline (百分点):")
        lines.append(f" {'Model':15s} | {'ΔR1':>8s} | {'ΔR2':>8s} | {'ΔRL':>8s} | {'ΔCos':>8s}")
        lines.append(f" {'-'*15}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        for cell_name in ['DCR Only', 'PSAL Only', 'Full']:
            if cell_name not in data:
                continue
            be = data[cell_name].get('best_eval') or data[cell_name].get('best_train', {})
            deltas = {}
            for m in metrics:
                d = (be.get(m, 0) - baseline_metrics.get(m, 0)) * 100
                sign = '+' if d > 0 else ''
                deltas[m] = f"{sign}{d:.2f}%"
            lines.append(
                f" {cell_name:15s} | {deltas['R1']:>8s} | {deltas['R2']:>8s} | "
                f"{deltas['RL']:>8s} | {deltas['Cos']:>8s}"
            )
    
    lines.append(f"{'=' * 80}")
    return '\n'.join(lines)


def generate_markdown_report(full_data, data_20pct, alignment_full, alignment_20pct, save_dir):
    """Generate a comprehensive markdown report."""
    report = []
    report.append("# BLiSS 消融实验统一评估报告\n")
    report.append(f"> 生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Full data table
    report.append("## 全量数据 (100%) — 2×2 消融表\n")
    report.append("| Model | DCR | PSAL | R1 | R2 | RL | Cos | Causal |")
    report.append("|-------|-----|------|----|----|----|----|--------|")
    
    for cell_name in ['Baseline', 'DCR Only', 'PSAL Only', 'Full']:
        if cell_name not in full_data:
            report.append(f"| {cell_name} | — | — | N/A | N/A | N/A | N/A | N/A |")
            continue
        parsed = full_data[cell_name]
        be = parsed.get('best_eval') or parsed.get('best_train', {})
        dcr = '✅' if parsed['DCR'] else '❌'
        psal = '✅' if parsed['PSAL'] else '❌'
        causal = f"{parsed.get('causal_final', 0):.4f}"
        report.append(
            f"| {cell_name} | {dcr} | {psal} | "
            f"{be.get('R1',0):.4f} | {be.get('R2',0):.4f} | "
            f"{be.get('RL',0):.4f} | {be.get('Cos',0):.4f} | {causal} |"
        )
    
    report.append("")
    
    # 20% data table
    report.append("## 20% 数据 — 2×2 消融表\n")
    report.append("| Model | DCR | PSAL | R1 | R2 | RL | Cos | Causal |")
    report.append("|-------|-----|------|----|----|----|----|--------|")
    
    for cell_name in ['Baseline', 'DCR Only', 'PSAL Only', 'Full']:
        if cell_name not in data_20pct:
            report.append(f"| {cell_name} | — | — | N/A | N/A | N/A | N/A | N/A |")
            continue
        parsed = data_20pct[cell_name]
        be = parsed.get('best_eval') or parsed.get('best_train', {})
        dcr = '✅' if parsed['DCR'] else '❌'
        psal = '✅' if parsed['PSAL'] else '❌'
        causal = f"{parsed.get('causal_final', 0):.4f}"
        report.append(
            f"| {cell_name} | {dcr} | {psal} | "
            f"{be.get('R1',0):.4f} | {be.get('R2',0):.4f} | "
            f"{be.get('RL',0):.4f} | {be.get('Cos',0):.4f} | {causal} |"
        )
    
    report.append("")
    
    # Delta analysis
    report.append("## Δ 增量贡献分析\n")
    for regime_name, data in [('全量数据', full_data), ('20% 数据', data_20pct)]:
        if 'Baseline' not in data:
            continue
        baseline = data['Baseline'].get('best_eval') or data['Baseline'].get('best_train', {})
        report.append(f"### {regime_name}\n")
        report.append("| Innovation | ΔR1 | ΔR2 | ΔRL | ΔCos |")
        report.append("|-----------|-----|-----|-----|------|")
        for cell_name in ['DCR Only', 'PSAL Only', 'Full']:
            if cell_name not in data:
                continue
            be = data[cell_name].get('best_eval') or data[cell_name].get('best_train', {})
            dr1 = (be.get('R1', 0) - baseline.get('R1', 0)) * 100
            dr2 = (be.get('R2', 0) - baseline.get('R2', 0)) * 100
            drl = (be.get('RL', 0) - baseline.get('RL', 0)) * 100
            dcos = (be.get('Cos', 0) - baseline.get('Cos', 0)) * 100
            fmt = lambda v: f"+{v:.2f}%" if v > 0 else f"{v:.2f}%"
            report.append(f"| {cell_name} | {fmt(dr1)} | {fmt(dr2)} | {fmt(drl)} | {fmt(dcos)} |")
        report.append("")
    
    # Alignment metrics (if available)
    if alignment_full or alignment_20pct:
        report.append("## 对齐质量评估\n")
        for regime_name, align_data in [('全量数据', alignment_full), ('20% 数据', alignment_20pct)]:
            if not align_data:
                continue
            report.append(f"### {regime_name}\n")
            report.append("| Model | AUC-PR | F1@p95 | Top-K Hit | Causal Sens. | N |")
            report.append("|-------|--------|--------|-----------|-------------|---|")
            for cell_name in ['Baseline', 'DCR Only', 'PSAL Only', 'Full']:
                if cell_name not in align_data:
                    continue
                m = align_data[cell_name]
                report.append(
                    f"| {cell_name} | {m.get('auc_pr_mean',0):.4f}±{m.get('auc_pr_std',0):.4f} | "
                    f"{m.get('f1_at_fixed_mean',0):.4f}±{m.get('f1_at_fixed_std',0):.4f} | "
                    f"{m.get('topk_hit_rate_mean',0):.4f}±{m.get('topk_hit_rate_std',0):.4f} | "
                    f"{m.get('causal_sensitivity_mean',0):.4f}±{m.get('causal_sensitivity_std',0):.4f} | "
                    f"{m.get('n_samples',0)} |"
                )
            report.append("")
    
    # Visualizations
    report.append("## 可视化\n")
    report.append("![2×2 消融柱状图](ablation_bar_chart.png)\n")
    report.append("![训练曲线](training_curves.png)\n")
    report.append("![因果损失曲线](causal_loss_curves.png)\n")
    report.append("![增量贡献热力图](delta_heatmap.png)\n")
    if alignment_full or alignment_20pct:
        report.append("![对齐雷达图](alignment_radar.png)\n")
    
    # Key findings
    report.append("## 关键发现\n")
    
    # Auto-detect key findings
    for regime_name, data in [('全量数据', full_data), ('20% 数据', data_20pct)]:
        if 'Baseline' not in data or len(data) < 2:
            continue
        baseline = data['Baseline'].get('best_eval') or data['Baseline'].get('best_train', {})
        best_r1_name = max(data.keys(), key=lambda k: (data[k].get('best_eval') or data[k].get('best_train', {})).get('R1', 0))
        best_r1 = (data[best_r1_name].get('best_eval') or data[best_r1_name].get('best_train', {})).get('R1', 0)
        delta = (best_r1 - baseline.get('R1', 0)) * 100
        report.append(f"- **{regime_name}**: 最优模型 = **{best_r1_name}** (R1={best_r1:.4f}, Δ={delta:+.2f}%p vs Baseline)")
    
    # DCR working check
    for data in [full_data, data_20pct]:
        if 'DCR Only' in data:
            cf = data['DCR Only'].get('causal_final', 0)
            if cf > 0:
                report.append(f"- ✅ DCR 修复生效: Causal Loss = {cf:.4f} (非零)")
            else:
                report.append(f"- ⚠️ DCR Causal Loss 仍为 0, 请检查")
            break
    
    report.append("")
    
    # Save
    md_path = os.path.join(save_dir, 'ablation_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"  ✓ 保存: {md_path}")
    return md_path


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='BLiSS 统一消融实验评估 v1.0')
    parser.add_argument('--skip_alignment', action='store_true',
                        help='跳过模型级别对齐评估 (仅解析日志)')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='对齐评估样本数 (默认50)')
    parser.add_argument('--save_dir', type=str,
                        default='./ablation_results',
                        help='结果保存目录')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(" BLiSS 统一消融实验评估  v1.0")
    print("=" * 70)

    # ---- Part 1: Log Parsing ----
    print("\n" + "─" * 50)
    print(" Part 1: 日志解析")
    print("─" * 50)

    print("\n[全量数据] 解析模型日志...")
    full_data = collect_all_logs(FULL_DATA_GRID, BLISS_LOG_DIR)

    print("\n[20% 数据] 解析模型日志...")
    data_20pct = collect_all_logs(DATA_20PCT_GRID, BLISS_20PCT_DIR)

    # Print summary tables
    table_full = build_summary_table(full_data, '全量数据 (100%) — 消融结果')
    table_20pct = build_summary_table(data_20pct, '20% 数据 — 消融结果')
    print(table_full)
    print(table_20pct)

    # ---- Part 2: Alignment Evaluation (optional) ----
    alignment_full = {}
    alignment_20pct = {}

    if not args.skip_alignment:
        print("\n" + "─" * 50)
        print(" Part 2: 对齐评估 (模型级别)")
        print("─" * 50)

        print("\n[全量数据] 对齐评估...")
        alignment_full = run_alignment_evaluation(
            FULL_DATA_GRID, BLISS_LOG_DIR, args.num_samples)

        print("\n[20% 数据] 对齐评估...")
        alignment_20pct = run_alignment_evaluation(
            DATA_20PCT_GRID, BLISS_20PCT_DIR, args.num_samples)
    else:
        print("\n⏭  跳过对齐评估 (使用 --skip_alignment)")

    # ---- Part 3: Visualization ----
    print("\n" + "─" * 50)
    print(" Part 3: 生成可视化")
    print("─" * 50)

    create_ablation_bar_chart(full_data, data_20pct, args.save_dir)
    create_training_curves(full_data, data_20pct, args.save_dir)
    create_causal_loss_chart(full_data, data_20pct, args.save_dir)
    create_delta_heatmap(full_data, data_20pct, args.save_dir)

    if alignment_full or alignment_20pct:
        create_alignment_radar(alignment_full, alignment_20pct, args.save_dir)

    # ---- Part 4: Report ----
    print("\n" + "─" * 50)
    print(" Part 4: 生成报告")
    print("─" * 50)

    generate_markdown_report(full_data, data_20pct, alignment_full, alignment_20pct, args.save_dir)

    # Save JSON
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    json_output = {
        '_metadata': {
            'script': 'evaluate_ablation_unified.py',
            'version': '1.0',
            'timestamp': datetime.datetime.now().isoformat(),
            'skip_alignment': args.skip_alignment,
            'num_samples': args.num_samples,
        },
        'full_data': {
            name: {
                'best_eval': p.get('best_eval', {}),
                'best_train': p.get('best_train', {}),
                'DCR': p['DCR'],
                'PSAL': p['PSAL'],
                'causal_final': p.get('causal_final', 0),
                'causal_mean': p.get('causal_mean', 0),
                'num_train_epochs': len(p['train_epochs']),
                'num_eval_epochs': len(p['eval_epochs']),
            }
            for name, p in full_data.items()
        },
        'data_20pct': {
            name: {
                'best_eval': p.get('best_eval', {}),
                'best_train': p.get('best_train', {}),
                'DCR': p['DCR'],
                'PSAL': p['PSAL'],
                'causal_final': p.get('causal_final', 0),
                'causal_mean': p.get('causal_mean', 0),
                'num_train_epochs': len(p['train_epochs']),
                'num_eval_epochs': len(p['eval_epochs']),
            }
            for name, p in data_20pct.items()
        },
    }

    if alignment_full:
        json_output['alignment_full'] = alignment_full
    if alignment_20pct:
        json_output['alignment_20pct'] = alignment_20pct

    json_path = os.path.join(args.save_dir, 'ablation_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(make_serializable(json_output), f, indent=2, ensure_ascii=False)
    print(f"  ✓ 保存: {json_path}")

    # Final summary
    print("\n" + "=" * 70)
    print(f" ✓ 所有结果已保存至: {args.save_dir}")
    print(f"   - ablation_bar_chart.png      2×2 消融柱状图")
    print(f"   - training_curves.png         训练曲线")
    print(f"   - causal_loss_curves.png      因果损失曲线")
    print(f"   - delta_heatmap.png           增量贡献热力图")
    if alignment_full or alignment_20pct:
        print(f"   - alignment_radar.png         对齐雷达图")
    print(f"   - ablation_report.md          完整 Markdown 报告")
    print(f"   - ablation_summary.json       机器可读结果")
    print("=" * 70)


if __name__ == '__main__':
    main()
