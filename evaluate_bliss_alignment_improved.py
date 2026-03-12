"""
BLiSS 改进对齐评估实验脚本  v2.0

主报指标（论文正文）:
  1. AUC-PR          — 无阈值依赖
  2. F1@fixed-rule   — 95th-percentile 自适应阈值（固定规则，非 oracle）
  3. Top-K Hit Rate  — 自适应 K
  4. Causal Sensitivity — 干预效应-GT帧数 Spearman 相关

附录指标:
  5. Best-F1 / Best-IoU（oracle 阈值，仅附录）

所有指标附 bootstrap 95% CI（5000 resamples, seed=42）。
详见 EVAL_SPEC.md。

用法:
    python evaluate_bliss_alignment_improved.py --num_samples 100
    python evaluate_bliss_alignment_improved.py --num_samples 10   # 快速测试
"""

import os
import sys
import json
import argparse
import hashlib
import datetime
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import precision_recall_curve, auc
from tqdm import tqdm

# ===========================================================================
# Determinism & Global Seed
# ===========================================================================
GLOBAL_SEED = 42
SCRIPT_VERSION = "2.0"

def set_global_seed(seed: int = GLOBAL_SEED):
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import build_args
from models import Model_BLiSS
from datasets import BLiSSDataset


# ============================================================================
# Model Loading (reused from bliss_visualization_comparison.py)
# ============================================================================

def load_model(model_dir, args):
    """加载模型"""
    model = Model_BLiSS(args=args)
    
    checkpoint_paths = [
        os.path.join(model_dir, 'checkpoint', 'model_best_text.pt'),
        os.path.join(model_dir, 'model_best_text.pt'),
        os.path.join(model_dir, 'checkpoint', 'model_best_video.pt'),
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    print(f"  ✓ 模型加载: {checkpoint_path}")
    return model


# ============================================================================
# Causal Matrix Computation (reused from bliss_visualization_comparison.py)
# ============================================================================

def compute_causal_matrix(model, sample, args):
    """计算因果效应矩阵（与原始脚本一致）"""
    video, video_summ, text, mask_video, mask_video_summ, mask_text, \
        video_label, text_label, sentence, highlight, \
        video_to_text_mask, text_to_video_mask = sample
    
    num_frames = video.shape[0]
    num_sentences = text.shape[0]
    
    video = video.unsqueeze(0).to(args.device)
    text = text.unsqueeze(0).to(args.device)
    mask_video = mask_video.unsqueeze(0).to(args.device)
    mask_text = mask_text.unsqueeze(0).to(args.device)
    video_label = video_label.unsqueeze(0).to(args.device)
    text_label = text_label.unsqueeze(0).to(args.device)
    video_to_text_mask_list = [video_to_text_mask.to(args.device)]
    text_to_video_mask_list = [text_to_video_mask.to(args.device)]
    
    gt_alignment = video_to_text_mask.cpu().numpy()
    
    with torch.no_grad():
        pred_video_orig, pred_text_orig, _ = model(
            video=video, text=text,
            mask_video=mask_video, mask_text=mask_text,
            video_label=video_label, text_label=text_label,
            video_to_text_mask_list=video_to_text_mask_list,
            text_to_video_mask_list=text_to_video_mask_list
        )
        
        causal_matrix = torch.zeros(num_frames, num_sentences)
        sentence_effects = []  # 每个句子的总干预效应
        
        for sent_idx in range(num_sentences):
            text_masked = text.clone()
            text_masked[0, sent_idx, :] = 0
            
            pred_video_masked, _, _ = model(
                video=video, text=text_masked,
                mask_video=mask_video, mask_text=mask_text,
                video_label=video_label, text_label=text_label,
                video_to_text_mask_list=video_to_text_mask_list,
                text_to_video_mask_list=text_to_video_mask_list
            )
            
            effect = (torch.sigmoid(pred_video_orig) - torch.sigmoid(pred_video_masked)).abs().squeeze()
            causal_matrix[:, sent_idx] = effect[:num_frames].cpu()
            sentence_effects.append(effect[:num_frames].sum().item())
    
    causal_np = causal_matrix.numpy()
    
    # Normalize to [0, 1]
    if causal_np.max() > 0:
        causal_np = causal_np / causal_np.max()
    
    # Trim to GT shape
    causal_np = causal_np[:gt_alignment.shape[0], :gt_alignment.shape[1]]
    
    return causal_np, gt_alignment, np.array(sentence_effects)


# ============================================================================
# Improved Metrics
# ============================================================================

THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]
FIXED_THRESHOLD_PERCENTILE = 95   # percentile of causal values used as fixed-rule threshold


def compute_precision_recall_f1(causal_matrix, gt_alignment, threshold):
    """在给定阈值下计算 Precision, Recall, F1"""
    pred_binary = (causal_matrix >= threshold).astype(float)
    gt_binary = (gt_alignment > 0).astype(float)
    
    tp = np.sum(pred_binary * gt_binary)
    fp = np.sum(pred_binary * (1 - gt_binary))
    fn = np.sum((1 - pred_binary) * gt_binary)
    
    precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def compute_iou(causal_matrix, gt_alignment, threshold):
    """在给定阈值下计算 IoU"""
    pred_binary = (causal_matrix >= threshold).astype(float)
    gt_binary = (gt_alignment > 0).astype(float)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(np.maximum(pred_binary, gt_binary))
    
    return intersection / (union + 1e-8) if union > 0 else 0.0


def compute_topk_hit_rate(causal_matrix, gt_alignment):
    """
    Top-K Hit Rate: K = GT中非零元素数量
    命中率 = |Top-K ∩ GT非零| / K
    """
    gt_binary = (gt_alignment > 0).astype(float)
    k = int(gt_binary.sum())
    
    if k == 0:
        return np.nan
    
    causal_flat = causal_matrix.flatten()
    gt_flat = gt_binary.flatten()
    
    # 取因果矩阵中最大的K个位置
    topk_indices = np.argsort(causal_flat)[-k:]
    
    # 计算有多少位于GT非零区域
    hits = gt_flat[topk_indices].sum()
    hit_rate = hits / k
    
    return hit_rate


def compute_auc_pr(causal_matrix, gt_alignment):
    """计算 Precision-Recall 曲线下面积 (AUC-PR)"""
    gt_flat = (gt_alignment > 0).astype(int).flatten()
    pred_flat = causal_matrix.flatten()
    
    # 需要至少有正例和负例
    if gt_flat.sum() == 0 or gt_flat.sum() == len(gt_flat):
        return np.nan
    
    precision_vals, recall_vals, _ = precision_recall_curve(gt_flat, pred_flat)
    auc_pr = auc(recall_vals, precision_vals)
    
    return auc_pr


def compute_causal_sensitivity(sentence_effects, gt_alignment):
    """
    因果敏感度：每个句子的干预效应总量与该句子在GT中对应的帧数之间的相关性
    高相关 = 模型对重要句子更敏感 = 对齐质量高
    """
    gt_binary = (gt_alignment > 0).astype(float)
    gt_frame_counts = gt_binary.sum(axis=0)  # 每个句子对应的帧数
    
    n_sentences = min(len(sentence_effects), len(gt_frame_counts))
    effects = sentence_effects[:n_sentences]
    counts = gt_frame_counts[:n_sentences]
    
    if n_sentences < 2 or np.std(effects) == 0 or np.std(counts) == 0:
        return np.nan
    
    corr, _ = stats.spearmanr(effects, counts)
    return corr


def compute_improved_metrics(causal_matrix, gt_alignment, sentence_effects):
    """计算所有改进指标"""
    results = {}
    
    # --- 1. Multi-threshold Precision/Recall/F1 ---
    best_f1 = 0
    best_threshold = 0
    threshold_results = {}
    
    for t in THRESHOLDS:
        p, r, f1 = compute_precision_recall_f1(causal_matrix, gt_alignment, t)
        iou = compute_iou(causal_matrix, gt_alignment, t)
        threshold_results[str(t)] = {
            'precision': p, 'recall': r, 'f1': f1, 'iou': iou
        }
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    results['threshold_analysis'] = threshold_results
    results['best_f1'] = best_f1
    results['best_threshold'] = best_threshold
    results['best_precision'] = threshold_results[str(best_threshold)]['precision']
    results['best_recall'] = threshold_results[str(best_threshold)]['recall']
    results['best_iou'] = threshold_results[str(best_threshold)]['iou']
    
    # --- 2. F1 @ fixed-rule threshold (percentile-based, NOT oracle) ---
    fixed_threshold = np.percentile(causal_matrix, FIXED_THRESHOLD_PERCENTILE)
    p_fix, r_fix, f1_fix = compute_precision_recall_f1(causal_matrix, gt_alignment, fixed_threshold)
    iou_fix = compute_iou(causal_matrix, gt_alignment, fixed_threshold)
    results['f1_at_fixed'] = f1_fix
    results['precision_at_fixed'] = p_fix
    results['recall_at_fixed'] = r_fix
    results['iou_at_fixed'] = iou_fix
    results['fixed_threshold_value'] = float(fixed_threshold)
    
    # --- 3. F1 @ 0.05 (hard-coded reference threshold) ---
    results['f1_at_005'] = threshold_results['0.05']['f1']
    results['precision_at_005'] = threshold_results['0.05']['precision']
    results['recall_at_005'] = threshold_results['0.05']['recall']
    
    # --- 4. Top-K Hit Rate ---
    results['topk_hit_rate'] = compute_topk_hit_rate(causal_matrix, gt_alignment)
    
    # --- 5. AUC-PR ---
    results['auc_pr'] = compute_auc_pr(causal_matrix, gt_alignment)
    
    # --- 6. Causal Sensitivity ---
    results['causal_sensitivity'] = compute_causal_sensitivity(sentence_effects, gt_alignment)
    
    # --- 7. Legacy metrics for comparison ---
    gt_flat = gt_alignment.flatten()
    causal_flat = causal_matrix.flatten()
    if len(gt_flat) == len(causal_flat) and np.std(gt_flat) > 0 and np.std(causal_flat) > 0:
        results['pearson_gtc'] = np.corrcoef(gt_flat, causal_flat)[0, 1]
        results['spearman_gtc'], _ = stats.spearmanr(gt_flat, causal_flat)
    else:
        results['pearson_gtc'] = np.nan
        results['spearman_gtc'] = np.nan
    
    # Energy-in-mask
    mask = (gt_alignment > 0).astype(float)
    results['energy_in_mask'] = np.sum(mask * causal_matrix) / (np.sum(mask) + 1e-8)
    
    return results


# ============================================================================
# Full Evaluation Pipeline
# ============================================================================

def evaluate_all_models(test_set, model_dirs, args, num_samples):
    """评估所有模型"""
    all_results = {}
    
    for model_name, model_path in model_dirs.items():
        print(f"\n{'=' * 60}")
        print(f"评估模型: {model_name}")
        print("=" * 60)
        
        try:
            args.model_dir = model_path
            model = load_model(model_path, args)
        except Exception as e:
            print(f"  ⚠️ 模型加载失败: {e}")
            continue
        
        sample_metrics = []
        n = min(num_samples, len(test_set))
        
        for idx in tqdm(range(n), desc=f"  分析 {model_name}"):
            sample = test_set[idx]
            
            # 检查是否有有效对齐mask
            if sample[10].dim() < 2:
                continue
            
            try:
                causal_np, gt_alignment, sentence_effects = compute_causal_matrix(model, sample, args)
                
                if gt_alignment.sum() == 0:
                    continue
                
                metrics = compute_improved_metrics(causal_np, gt_alignment, sentence_effects)
                sample_metrics.append(metrics)
            except Exception as e:
                print(f"  ⚠️ 样本 {idx} 失败: {e}")
                continue
        
        if not sample_metrics:
            print(f"  ⚠️ 无有效样本")
            continue
        
        # Aggregate metrics
        aggregated = aggregate_metrics(sample_metrics)
        aggregated['n_samples'] = len(sample_metrics)
        all_results[model_name] = aggregated
        
        # Print summary — primary metrics first
        print(f"\n  --- {model_name} 汇总 (n={len(sample_metrics)}) ---")
        print(f"  [PRIMARY] AUC-PR:          {aggregated['auc_pr_mean']:.4f} ± {aggregated['auc_pr_std']:.4f}")
        print(f"  [PRIMARY] F1@fixed(p95):   {aggregated['f1_at_fixed_mean']:.4f} ± {aggregated['f1_at_fixed_std']:.4f}")
        print(f"  [PRIMARY] F1@0.05:         {aggregated['f1_at_005_mean']:.4f} ± {aggregated['f1_at_005_std']:.4f}")
        print(f"  [PRIMARY] Top-K Hit Rate:  {aggregated['topk_hit_rate_mean']:.4f} ± {aggregated['topk_hit_rate_std']:.4f}")
        print(f"  [PRIMARY] Causal Sens.:    {aggregated['causal_sensitivity_mean']:.4f} ± {aggregated['causal_sensitivity_std']:.4f}")
        print(f"  [APPENDIX] Best F1:        {aggregated['best_f1_mean']:.4f} ± {aggregated['best_f1_std']:.4f}")
        print(f"  [APPENDIX] Best IoU:       {aggregated['best_iou_mean']:.4f} ± {aggregated['best_iou_std']:.4f}")
        print(f"  [LEGACY]  Pearson:         {aggregated['pearson_gtc_mean']:.4f}")
        
        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_results


def aggregate_metrics(sample_metrics):
    """聚合多样本的指标（macro-average: 每样本一个值 → 取均值）"""
    # Scalar metrics to aggregate
    scalar_keys = [
        # Primary (paper body)
        'auc_pr', 'f1_at_fixed', 'f1_at_005', 'topk_hit_rate', 'causal_sensitivity',
        'precision_at_fixed', 'recall_at_fixed', 'iou_at_fixed',
        'precision_at_005', 'recall_at_005',
        'fixed_threshold_value',
        # Appendix (oracle)
        'best_f1', 'best_threshold', 'best_precision', 'best_recall', 'best_iou',
        # Legacy
        'pearson_gtc', 'spearman_gtc', 'energy_in_mask'
    ]
    
    result = {}
    for key in scalar_keys:
        values = [m[key] for m in sample_metrics if not np.isnan(m.get(key, np.nan))]
        if values:
            result[f'{key}_mean'] = float(np.mean(values))
            result[f'{key}_std'] = float(np.std(values))
            result[f'{key}_median'] = float(np.median(values))
        else:
            result[f'{key}_mean'] = float('nan')
            result[f'{key}_std'] = float('nan')
            result[f'{key}_median'] = float('nan')
    
    # Per-threshold aggregation
    threshold_agg = {}
    for t_str in [str(t) for t in THRESHOLDS]:
        for metric in ['precision', 'recall', 'f1', 'iou']:
            values = [m['threshold_analysis'][t_str][metric] 
                     for m in sample_metrics 
                     if t_str in m.get('threshold_analysis', {})]
            if values:
                threshold_agg.setdefault(t_str, {})[f'{metric}_mean'] = float(np.mean(values))
                threshold_agg.setdefault(t_str, {})[f'{metric}_std'] = float(np.std(values))
    
    result['threshold_analysis'] = threshold_agg
    
    # Bootstrap 95% CI for ALL scalar metrics (not just 3)
    # Use a local RNG to avoid polluting global state
    rng = np.random.RandomState(GLOBAL_SEED)
    ci_keys = [
        'auc_pr', 'f1_at_fixed', 'f1_at_005', 'topk_hit_rate', 'causal_sensitivity',
        'best_f1', 'best_iou',
    ]
    for key in ci_keys:
        values = [m[key] for m in sample_metrics if not np.isnan(m.get(key, np.nan))]
        if len(values) >= 10:
            boot_means = [float(np.mean(rng.choice(values, size=len(values), replace=True)))
                         for _ in range(5000)]
            result[f'{key}_ci_lower'] = float(np.percentile(boot_means, 2.5))
            result[f'{key}_ci_upper'] = float(np.percentile(boot_means, 97.5))
    
    return result


# ============================================================================
# Visualization
# ============================================================================

def visualize_results(all_results, save_dir):
    """生成对比可视化"""
    os.makedirs(save_dir, exist_ok=True)
    
    model_names = list(all_results.keys())
    short_names = [n.replace('BLiSS_', '') for n in model_names]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12'][:len(model_names)]
    
    # ========== 1. Multi-metric bar chart (PRIMARY → APPENDIX) ==========
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    metrics_to_plot = [
        # --- Primary (paper body) ---
        ('auc_pr_mean', 'auc_pr_std', 'AUC-PR  [PRIMARY]', 'Threshold-free, higher = better'),
        ('f1_at_fixed_mean', 'f1_at_fixed_std', 'F1 @ 95th-percentile  [PRIMARY]', 'Fixed-rule threshold, NOT oracle'),
        ('f1_at_005_mean', 'f1_at_005_std', 'F1 @ 0.05  [PRIMARY]', 'Hard-coded low threshold'),
        ('topk_hit_rate_mean', 'topk_hit_rate_std', 'Top-K Hit Rate  [PRIMARY]', 'K = #GT positives'),
        ('causal_sensitivity_mean', 'causal_sensitivity_std', 'Causal Sensitivity  [PRIMARY]', 'Spearman(effect, GT frames)'),
        # --- Appendix ---
        ('best_f1_mean', 'best_f1_std', 'Best F1  [APPENDIX]', 'Oracle threshold — for reference only'),
    ]
    
    for ax, (mean_key, std_key, title, subtitle) in zip(axes.flatten(), metrics_to_plot):
        means = [all_results[m].get(mean_key, 0) for m in model_names]
        stds = [all_results[m].get(std_key, 0) for m in model_names]
        
        bars = ax.bar(short_names, means, yerr=stds, color=colors, alpha=0.75,
                      capsize=5, edgecolor='black', linewidth=0.5)
        # Grey out appendix panel title
        title_color = '#333333' if '[PRIMARY]' in title else '#888888'
        ax.set_title(f'{title}\n({subtitle})', fontsize=11, fontweight='bold', color=title_color)
        ax.tick_params(axis='x', rotation=20, labelsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Value labels
        for bar, val in zip(bars, means):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('BLiSS Alignment Evaluation — Primary Metrics + Appendix', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path1 = os.path.join(save_dir, 'improved_metrics_comparison.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {path1}")
    
    # ========== 2. Threshold analysis curves ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (model_name, color, short) in enumerate(zip(model_names, colors, short_names)):
        data = all_results[model_name].get('threshold_analysis', {})
        thresholds = sorted([float(t) for t in data.keys()])
        
        if not thresholds:
            continue
        
        f1_means = [data[str(t)].get('f1_mean', 0) for t in thresholds]
        iou_means = [data[str(t)].get('iou_mean', 0) for t in thresholds]
        prec_means = [data[str(t)].get('precision_mean', 0) for t in thresholds]
        recall_means = [data[str(t)].get('recall_mean', 0) for t in thresholds]
        
        axes[0].plot(thresholds, f1_means, '-o', color=color, label=short, linewidth=2, markersize=5)
        axes[1].plot(thresholds, iou_means, '-o', color=color, label=short, linewidth=2, markersize=5)
        axes[2].plot(thresholds, prec_means, '-', color=color, label=f'{short} (P)', linewidth=2)
        axes[2].plot(thresholds, recall_means, '--', color=color, label=f'{short} (R)', linewidth=1.5)
    
    axes[0].set_title('F1 Score vs. Threshold', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Binarization Threshold')
    axes[0].set_ylabel('F1 Score')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].set_title('IoU vs. Threshold', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Binarization Threshold')
    axes[1].set_ylabel('IoU')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    axes[2].set_title('Precision & Recall vs. Threshold', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Binarization Threshold')
    axes[2].set_ylabel('Value')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    
    plt.suptitle('Threshold Analysis: How Binarization Threshold Affects Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path2 = os.path.join(save_dir, 'threshold_analysis.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {path2}")
    
    # ========== 3. Radar chart — PRIMARY metrics only ==========
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    radar_metrics = ['auc_pr_mean', 'f1_at_fixed_mean', 'f1_at_005_mean', 'topk_hit_rate_mean', 'causal_sensitivity_mean']
    radar_labels = ['AUC-PR', 'F1@p95', 'F1@0.05', 'Top-K Hit', 'Causal Sens.']
    
    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]  # Close polygon
    
    for i, (model_name, color, short) in enumerate(zip(model_names, colors, short_names)):
        values = []
        for m in radar_metrics:
            v = all_results[model_name].get(m, 0)
            values.append(v if not np.isnan(v) else 0)
        values += values[:1]
        
        ax.plot(angles, values, '-o', color=color, linewidth=2, label=short, markersize=5)
        ax.fill(angles, values, color=color, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=11)
    ax.set_title('Primary Metrics Radar Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    path3 = os.path.join(save_dir, 'radar_comparison.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {path3}")


def _fmt_ci(data, key):
    """Format value ± std [CI_low, CI_high] for display."""
    mean = data.get(f'{key}_mean', float('nan'))
    ci_lo = data.get(f'{key}_ci_lower', None)
    ci_hi = data.get(f'{key}_ci_upper', None)
    if ci_lo is not None and ci_hi is not None:
        return f"{mean:.4f} [{ci_lo:.4f},{ci_hi:.4f}]"
    return f"{mean:.4f}"


def print_summary_table(all_results):
    """打印汇总表 — primary metrics first, appendix at end."""
    print("\n" + "=" * 140)
    print("改进对齐评估 — 汇总表  (PRIMARY → APPENDIX)")
    print("=" * 140)
    
    # --- Primary metrics table ---
    headers = ['Model', 'AUC-PR', 'F1@p95', 'F1@0.05', 'Top-K Hit', 'Causal Sens.', 'Best F1[app]', 'Best IoU[app]', 'N']
    col_widths = [38, 26, 26, 26, 26, 14, 14, 14, 6]
    
    header_str = ' | '.join(h.center(w) for h, w in zip(headers, col_widths))
    print(header_str)
    print('-' * len(header_str))
    
    for model_name, data in all_results.items():
        short = model_name.replace('BLiSS_', '')
        vals = [
            _fmt_ci(data, 'auc_pr'),
            _fmt_ci(data, 'f1_at_fixed'),
            _fmt_ci(data, 'f1_at_005'),
            _fmt_ci(data, 'topk_hit_rate'),
            f"{data.get('causal_sensitivity_mean', 0):.4f}",
            f"{data.get('best_f1_mean', 0):.4f}",
            f"{data.get('best_iou_mean', 0):.4f}",
            str(data.get('n_samples', 0)),
        ]
        row = [short.ljust(col_widths[0])] + [v.center(w) for v, w in zip(vals, col_widths[1:])]
        print(' | '.join(row))
    
    print("=" * 140)
    
    # Highlight best model per primary metric
    print("\n🏆 Primary 指标最优模型:")
    metric_keys = {
        'AUC-PR': 'auc_pr_mean',
        'F1@fixed(p95)': 'f1_at_fixed_mean',
        'F1@0.05': 'f1_at_005_mean',
        'Top-K Hit Rate': 'topk_hit_rate_mean',
        'Causal Sensitivity': 'causal_sensitivity_mean',
    }
    for display, key in metric_keys.items():
        best_model = max(all_results.keys(), 
                        key=lambda m: all_results[m].get(key, -float('inf')))
        best_val = all_results[best_model].get(key, 0)
        print(f"  {display:>20s}: {best_model.replace('BLiSS_', '')} ({best_val:.4f})")


# ============================================================================
# Main
# ============================================================================

def _build_metadata():
    """Build metadata block for JSON output reproducibility."""
    meta = {
        'script': 'evaluate_bliss_alignment_improved.py',
        'version': SCRIPT_VERSION,
        'seed': GLOBAL_SEED,
        'timestamp': datetime.datetime.now().astimezone().isoformat(),
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'primary_metrics': ['auc_pr', 'f1_at_fixed', 'f1_at_005', 'topk_hit_rate', 'causal_sensitivity'],
        'appendix_metrics': ['best_f1', 'best_iou'],
        'bootstrap_resamples': 5000,
        'fixed_threshold_rule': f'{FIXED_THRESHOLD_PERCENTILE}th percentile of causal matrix',
        'averaging': 'macro (per-sample then mean)',
        'normalization': 'per-sample global-max to [0,1]',
        'gt_binarization': '> 0',
        'threshold_inclusion': '>= (inclusive)',
    }
    # Attempt to capture git commit
    try:
        import subprocess
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL
        ).decode().strip()
        meta['git_commit'] = commit
    except Exception:
        meta['git_commit'] = 'unavailable'
    return meta


def main():
    parser = argparse.ArgumentParser(description='BLiSS 改进对齐评估实验 v2.0')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='分析的测试样本数量 (默认100, 快速测试用10)')
    parser.add_argument('--save_dir', type=str, 
                        default='./interpretability_results/bliss_improved_eval',
                        help='结果保存目录')
    cli_args = parser.parse_args()
    
    # ---- Determinism ----
    set_global_seed(GLOBAL_SEED)
    
    print("\n" + "=" * 70)
    print(f"BLiSS 改进对齐评估实验 v{SCRIPT_VERSION}  (seed={GLOBAL_SEED})")
    print("=" * 70)
    
    # Build args
    old_argv = sys.argv
    sys.argv = ['', '--dataset=BLiSS']
    args = build_args()
    sys.argv = old_argv
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"设备: {args.device}")
    print(f"样本数: {cli_args.num_samples}")
    
    # Load dataset
    print("\n加载 BLiSS 测试集...")
    test_set = BLiSSDataset(mode='test', args=args)
    print(f"  ✓ 共 {len(test_set)} 个测试样本")
    
    # Discover models
    bliss_log_dir = os.path.join(os.path.dirname(__file__), 'logs', 'BLiss')
    model_dirs = {}
    for name in sorted(os.listdir(bliss_log_dir)):
        model_path = os.path.join(bliss_log_dir, name)
        if os.path.isdir(model_path):
            model_dirs[name] = model_path
    
    print(f"\n发现 {len(model_dirs)} 个模型:")
    for name in model_dirs:
        print(f"  • {name}")
    
    # Evaluate
    all_results = evaluate_all_models(test_set, model_dirs, args, cli_args.num_samples)
    
    if not all_results:
        print("\n⚠️ 无有效结果")
        return
    
    # Visualize
    print("\n" + "=" * 70)
    print("生成可视化...")
    print("=" * 70)
    visualize_results(all_results, cli_args.save_dir)
    
    # Summary table
    print_summary_table(all_results)
    
    # Save JSON with metadata
    json_path = os.path.join(cli_args.save_dir, 'improved_alignment_results.json')
    
    def convert_nan(obj):
        if isinstance(obj, dict):
            return {k: convert_nan(v) for k, v in obj.items()}
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj
    
    output = {
        '_metadata': _build_metadata(),
        **all_results
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(convert_nan(output), f, indent=2, ensure_ascii=False)
    print(f"\n✓ 详细结果保存至: {json_path}")
    
    print("\n" + "=" * 70)
    print(f"所有结果已保存至: {cli_args.save_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
