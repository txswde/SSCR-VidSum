"""
BLiSS Enhanced Visualization Comparison - 可视化对比脚本

生成三种关键可视化:
1. 对齐矩阵热力图对比：有监督 vs 无监督 attention matrix
2. GT相关性分布图：展示多个样本上的相关性分布
3. 因果干预效应图：展示特定样本上移除不同句子的效应差异
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互模式
from tqdm import tqdm
import seaborn as sns
from scipy import stats  # For Spearman correlation and t-test

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import build_args
from models import Model_BLiSS
from datasets import BLiSSDataset

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_model(model_dir, args):
    """加载模型"""
    model = Model_BLiSS(args=args)
    
    # 尝试多种checkpoint路径
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
    
    return model


def compute_causal_matrix(model, sample, args):
    """计算因果效应矩阵"""
    video, video_summ, text, mask_video, mask_video_summ, mask_text, \
        video_label, text_label, sentence, highlight, \
        video_to_text_mask, text_to_video_mask = sample
    
    num_frames = video.shape[0]
    num_sentences = text.shape[0]
    
    # Create batch
    video = video.unsqueeze(0).to(args.device)
    text = text.unsqueeze(0).to(args.device)
    mask_video = mask_video.unsqueeze(0).to(args.device)
    mask_text = mask_text.unsqueeze(0).to(args.device)
    video_label = video_label.unsqueeze(0).to(args.device)
    text_label = text_label.unsqueeze(0).to(args.device)
    video_to_text_mask_list = [video_to_text_mask.to(args.device)]
    text_to_video_mask_list = [text_to_video_mask.to(args.device)]
    
    with torch.no_grad():
        # Original prediction
        pred_video_orig, pred_text_orig, _ = model(
            video=video, text=text,
            mask_video=mask_video, mask_text=mask_text,
            video_label=video_label, text_label=text_label,
            video_to_text_mask_list=video_to_text_mask_list,
            text_to_video_mask_list=text_to_video_mask_list
        )
        
        # Compute causal effect matrix
        causal_matrix = torch.zeros(num_frames, num_sentences)
        intervention_effects = []  # 存储每个句子的干预效应
        
        for sent_idx in range(num_sentences):
            # Mask out sentence sent_idx
            text_masked = text.clone()
            text_masked[0, sent_idx, :] = 0  # Zero out the sentence
            
            # Get prediction with masked text
            pred_video_masked, _, _ = model(
                video=video, text=text_masked,
                mask_video=mask_video, mask_text=mask_text,
                video_label=video_label, text_label=text_label,
                video_to_text_mask_list=video_to_text_mask_list,
                text_to_video_mask_list=text_to_video_mask_list
            )
            
            # Causal effect = change in prediction
            effect = (torch.sigmoid(pred_video_orig) - torch.sigmoid(pred_video_masked)).abs().squeeze()
            causal_matrix[:, sent_idx] = effect[:num_frames].cpu()
            intervention_effects.append({
                'sentence_idx': sent_idx,
                'effect': effect[:num_frames].cpu().numpy(),
                'total_effect': effect[:num_frames].sum().item()
            })
    
    # Get GT alignment
    gt_alignment = video_to_text_mask.cpu().numpy()
    causal_np = causal_matrix.numpy()
    
    # Normalize
    if causal_np.max() > 0:
        causal_np = causal_np / causal_np.max()
    
    # Calculate correlation
    spearman_correlation = np.nan
    energy_in_mask = np.nan
    if gt_alignment.sum() > 0:
        gt_flat = gt_alignment.flatten()
        causal_flat = causal_np[:gt_alignment.shape[0], :gt_alignment.shape[1]].flatten()
        if len(gt_flat) == len(causal_flat):
            correlation = np.corrcoef(gt_flat, causal_flat)[0, 1]
            # Spearman Correlation (rank-based, scale-invariant)
            spearman_correlation, _ = stats.spearmanr(gt_flat, causal_flat)
            # Energy-in-mask: sum(M * A) / sum(M)
            mask_aligned = gt_alignment.astype(float)
            causal_aligned = causal_np[:gt_alignment.shape[0], :gt_alignment.shape[1]]
            energy_in_mask = np.sum(mask_aligned * causal_aligned) / (np.sum(mask_aligned) + 1e-8)
        else:
            correlation = np.nan
    else:
        correlation = np.nan
    
    # Row vs Column variance ratio
    row_var = np.var(causal_np, axis=1).mean()
    col_var = np.var(causal_np, axis=0).mean()
    var_ratio = row_var / (col_var + 1e-8)
    
    return {
        'causal_matrix': causal_np[:gt_alignment.shape[0], :gt_alignment.shape[1]],
        'gt_alignment': gt_alignment,
        'correlation': correlation,
        'spearman_correlation': spearman_correlation,
        'energy_in_mask': energy_in_mask,
        'var_ratio': var_ratio,
        'intervention_effects': intervention_effects,
        'num_frames': num_frames,
        'num_sentences': num_sentences,
        'orig_pred': torch.sigmoid(pred_video_orig).squeeze().cpu().numpy()[:num_frames]
    }


def visualize_alignment_comparison(models_data, save_dir):
    """
    生成对齐矩阵热力图对比
    并排展示有监督vs无监督的attention matrix
    """
    print("\n[1/3] 生成对齐矩阵热力图对比...")
    
    # 选择前4个样本进行可视化
    sample_indices = list(range(min(4, len(list(models_data.values())[0]['samples']))))
    
    for sample_idx in sample_indices:
        num_models = len(models_data)
        fig, axes = plt.subplots(2, num_models + 1, figsize=(5 * (num_models + 1), 10))
        
        # 获取第一个模型的GT作为参考
        first_model_name = list(models_data.keys())[0]
        gt_alignment = models_data[first_model_name]['samples'][sample_idx]['gt_alignment']
        
        # 第一行：GT对齐 + 各模型的因果矩阵
        # GT
        if num_models > 0:
            ax_gt = axes[0, 0]
            im_gt = ax_gt.imshow(gt_alignment, aspect='auto', cmap='Blues', vmin=0, vmax=1)
            ax_gt.set_title('Ground Truth\nAlignment', fontsize=12, fontweight='bold')
            ax_gt.set_xlabel('Sentence Index')
            ax_gt.set_ylabel('Frame Index')
            plt.colorbar(im_gt, ax=ax_gt, fraction=0.046)
        
        # 各模型的因果矩阵
        for i, (model_name, model_data) in enumerate(models_data.items()):
            sample_data = model_data['samples'][sample_idx]
            causal_matrix = sample_data['causal_matrix']
            correlation = sample_data['correlation']
            
            ax = axes[0, i + 1]
            im = ax.imshow(causal_matrix, aspect='auto', cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'{model_name}\nCorr: {correlation:.3f}', fontsize=11)
            ax.set_xlabel('Sentence Index')
            if i == 0:
                ax.set_ylabel('Frame Index')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 第二行：差异图
        axes[1, 0].axis('off')
        
        for i, (model_name, model_data) in enumerate(models_data.items()):
            sample_data = model_data['samples'][sample_idx]
            causal_matrix = sample_data['causal_matrix']
            
            ax = axes[1, i + 1]
            # 计算与GT的差异
            diff = causal_matrix - gt_alignment.astype(float)
            im = ax.imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title(f'{model_name}\n(Difference from GT)', fontsize=11)
            ax.set_xlabel('Sentence Index')
            if i == 0:
                ax.set_ylabel('Frame Index')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.suptitle(f'Sample {sample_idx}: Alignment Matrix Comparison\n(Red=Overestimate, Blue=Underestimate)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        save_path = os.path.join(save_dir, f'alignment_comparison_sample_{sample_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: {save_path}")


def visualize_correlation_distribution(models_data, save_dir):
    """
    生成GT相关性分布图
    展示多个样本上的相关性分布
    """
    print("\n[2/3] 生成GT相关性分布图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # === 左图：箱线图对比 ===
    ax1 = axes[0]
    correlations_data = []
    model_names = []
    colors = []
    color_palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    for i, (model_name, model_data) in enumerate(models_data.items()):
        corrs = [s['correlation'] for s in model_data['samples'] if not np.isnan(s['correlation'])]
        if corrs:
            correlations_data.append(corrs)
            model_names.append(model_name.replace('BLiSS_', ''))
            colors.append(color_palette[i % len(color_palette)])
    
    bp = ax1.boxplot(correlations_data, labels=model_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax1.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='Good Threshold (0.3)')
    ax1.axhline(y=0.5, color='blue', linestyle='--', alpha=0.7, label='Excellent Threshold (0.5)')
    ax1.set_ylabel('Correlation with GT', fontsize=12)
    ax1.set_title('GT Correlation Distribution by Model', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(axis='y', alpha=0.3)
    
    # === 右图：核密度估计图 ===
    ax2 = axes[1]
    for i, (model_name, model_data) in enumerate(models_data.items()):
        corrs = [s['correlation'] for s in model_data['samples'] if not np.isnan(s['correlation'])]
        if corrs:
            short_name = model_name.replace('BLiSS_', '')
            avg_corr = np.mean(corrs)
            sns.kdeplot(corrs, ax=ax2, label=f'{short_name} (μ={avg_corr:.3f})', 
                       color=color_palette[i % len(color_palette)], linewidth=2)
    
    ax2.axvline(x=0.3, color='green', linestyle='--', alpha=0.7)
    ax2.axvline(x=0.5, color='blue', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Correlation with GT', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Correlation Distribution Density', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'correlation_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {save_path}")
    
    # === 额外：散点图对比 ===
    fig2, ax = plt.subplots(figsize=(10, 8))
    
    for i, (model_name, model_data) in enumerate(models_data.items()):
        corrs = [s['correlation'] for s in model_data['samples'] if not np.isnan(s['correlation'])]
        var_ratios = [s['var_ratio'] for s in model_data['samples'] if not np.isnan(s['correlation'])]
        short_name = model_name.replace('BLiSS_', '')
        
        ax.scatter(corrs, var_ratios, label=short_name, 
                  color=color_palette[i % len(color_palette)], 
                  alpha=0.6, s=80, edgecolors='white', linewidth=1)
        
        # 添加均值点
        if corrs and var_ratios:
            ax.scatter([np.mean(corrs)], [np.mean(var_ratios)], 
                      color=color_palette[i % len(color_palette)],
                      marker='*', s=300, edgecolors='black', linewidth=1.5)
    
    ax.axvline(x=0.3, color='green', linestyle='--', alpha=0.5, label='Corr=0.3')
    ax.axhline(y=0.5, color='blue', linestyle='--', alpha=0.5, label='VarRatio=0.5')
    ax.set_xlabel('GT Correlation (Higher = Better Alignment Discovery)', fontsize=12)
    ax.set_ylabel('Row/Col Variance Ratio (Higher = More Position Sensitive)', fontsize=12)
    ax.set_title('Model Comparison: Alignment Quality vs Position Sensitivity\n(Stars = Model Average)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    
    save_path2 = os.path.join(save_dir, 'scatter_comparison.png')
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {save_path2}")


def visualize_intervention_effects(models_data, save_dir):
    """
    生成因果干预效应图
    展示特定样本上移除不同句子的效应差异
    """
    print("\n[3/3] 生成因果干预效应图...")
    
    # 选择一个具有代表性的样本
    sample_idx = 0
    
    for model_name, model_data in models_data.items():
        sample_data = model_data['samples'][sample_idx]
        intervention_effects = sample_data['intervention_effects']
        gt_alignment = sample_data['gt_alignment']
        num_frames = sample_data['num_frames']
        num_sentences = sample_data['num_sentences']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # === 左上：干预效应热力图 ===
        ax1 = axes[0, 0]
        effect_matrix = sample_data['causal_matrix']
        im1 = ax1.imshow(effect_matrix, aspect='auto', cmap='hot')
        ax1.set_title(f'Causal Effect Matrix\n(Row i, Col j = Effect on Frame i when removing Sentence j)', fontsize=11)
        ax1.set_xlabel('Sentence Index')
        ax1.set_ylabel('Frame Index')
        plt.colorbar(im1, ax=ax1)
        
        # === 右上：GT对齐叠加 ===
        ax2 = axes[0, 1]
        # 叠加显示
        ax2.imshow(effect_matrix, aspect='auto', cmap='hot', alpha=0.7)
        ax2.contour(gt_alignment, levels=[0.5], colors='cyan', linewidths=2)
        ax2.set_title('Causal Effects with GT Alignment Overlay\n(Cyan contours = GT alignment boundaries)', fontsize=11)
        ax2.set_xlabel('Sentence Index')
        ax2.set_ylabel('Frame Index')
        
        # === 左下：各句子的总干预效应 ===
        ax3 = axes[1, 0]
        total_effects = [ie['total_effect'] for ie in intervention_effects]
        sentence_indices = list(range(num_sentences))
        
        # 计算GT中每个句子对应的帧数量
        gt_frame_counts = gt_alignment.sum(axis=0)
        
        bars = ax3.bar(sentence_indices, total_effects, color='coral', alpha=0.7, label='Total Causal Effect')
        ax3.plot(sentence_indices, gt_frame_counts * max(total_effects) / max(gt_frame_counts.max(), 1), 
                'g--', linewidth=2, marker='o', label='GT Frame Count (scaled)')
        ax3.set_xlabel('Sentence Index')
        ax3.set_ylabel('Total Effect Magnitude')
        ax3.set_title('Total Intervention Effect per Sentence\n(Higher = Sentence is more important for predictions)', fontsize=11)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # === 右下：选择特定句子展示帧级效应 ===
        ax4 = axes[1, 1]
        
        # 选择效应最大和最小的句子
        effects_sum = [ie['total_effect'] for ie in intervention_effects]
        max_idx = np.argmax(effects_sum)
        min_idx = np.argmin(effects_sum)
        
        frame_indices = list(range(num_frames))
        
        ax4.plot(frame_indices, intervention_effects[max_idx]['effect'], 
                'r-', linewidth=2, label=f'Sentence {max_idx} (Max Effect)')
        ax4.plot(frame_indices, intervention_effects[min_idx]['effect'], 
                'b-', linewidth=2, label=f'Sentence {min_idx} (Min Effect)')
        
        # 标记GT对齐区域
        for sent_idx, color in [(max_idx, 'red'), (min_idx, 'blue')]:
            aligned_frames = np.where(gt_alignment[:, sent_idx] > 0)[0]
            if len(aligned_frames) > 0:
                ax4.axvspan(aligned_frames[0], aligned_frames[-1], 
                           alpha=0.15, color=color, label=f'GT Region for Sent {sent_idx}')
        
        ax4.set_xlabel('Frame Index')
        ax4.set_ylabel('Effect Magnitude')
        ax4.set_title('Frame-level Intervention Effects\n(Shaded regions = GT alignment zones)', fontsize=11)
        ax4.legend(loc='upper right')
        ax4.grid(alpha=0.3)
        
        plt.suptitle(f'{model_name}: Causal Intervention Analysis (Sample {sample_idx})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        short_name = model_name.replace('BLiSS_', '').replace('/', '_')
        save_path = os.path.join(save_dir, f'intervention_effects_{short_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: {save_path}")


def generate_summary_figure(models_data, save_dir):
    """生成汇总对比图"""
    print("\n[额外] 生成汇总对比图...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    model_names_short = [name.replace('BLiSS_', '') for name in models_data.keys()]
    color_palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    # === 柱状图1：平均GT相关性 ===
    ax1 = axes[0]
    avg_corrs = []
    for model_name, model_data in models_data.items():
        corrs = [s['correlation'] for s in model_data['samples'] if not np.isnan(s['correlation'])]
        avg_corrs.append(np.mean(corrs) if corrs else 0)
    
    bars1 = ax1.bar(model_names_short, avg_corrs, color=color_palette[:len(avg_corrs)], alpha=0.7)
    ax1.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='Threshold')
    ax1.set_ylabel('Average GT Correlation', fontsize=12)
    ax1.set_title('GT Alignment Correlation\n(Higher = Discovered alignment matches GT)', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, val in zip(bars1, avg_corrs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # === 柱状图2：平均方差比 ===
    ax2 = axes[1]
    avg_var_ratios = []
    for model_name, model_data in models_data.items():
        ratios = [s['var_ratio'] for s in model_data['samples']]
        avg_var_ratios.append(np.mean(ratios) if ratios else 0)
    
    bars2 = ax2.bar(model_names_short, avg_var_ratios, color=color_palette[:len(avg_var_ratios)], alpha=0.7)
    ax2.axhline(y=0.5, color='blue', linestyle='--', alpha=0.7, label='Threshold')
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Ideal')
    ax2.set_ylabel('Average Row/Col Variance Ratio', fontsize=12)
    ax2.set_title('Position Sensitivity\n(≈1 = Local patterns, >>1 = Stripe patterns)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, avg_var_ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # === 样本数量 ===
    ax3 = axes[2]
    sample_counts = [len(model_data['samples']) for model_data in models_data.values()]
    
    bars3 = ax3.bar(model_names_short, sample_counts, color=color_palette[:len(sample_counts)], alpha=0.7)
    ax3.set_ylabel('Number of Samples', fontsize=12)
    ax3.set_title('Analyzed Samples', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=15)
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars3, sample_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('BLiSS Model Comparison Summary', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    save_path = os.path.join(save_dir, 'summary_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {save_path}")


def compute_statistical_significance(models_data, save_dir):
    """
    计算统计显著性指标:
    1. Bootstrap 95% CI for GTC
    2. Paired t-test between models
    """
    print("\n[统计显著性分析]")
    print("="*70)
    
    results = {}
    model_names = list(models_data.keys())
    
    for model_name, model_data in models_data.items():
        corrs = model_data.get('correlations', [])
        spearman_corrs = [s['spearman_correlation'] for s in model_data['samples'] 
                         if not np.isnan(s['spearman_correlation'])]
        energy_vals = [s['energy_in_mask'] for s in model_data['samples'] 
                      if not np.isnan(s['energy_in_mask'])]
        
        if len(corrs) < 2:
            print(f"  ⚠️ {model_name}: 样本不足，跳过")
            continue
        
        # Bootstrap 95% CI for Pearson GTC
        n_bootstrap = 10000
        bootstrap_means = []
        np.random.seed(42)
        for _ in range(n_bootstrap):
            resample = np.random.choice(corrs, size=len(corrs), replace=True)
            bootstrap_means.append(np.mean(resample))
        
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        mean_val = np.mean(corrs)
        
        # Bootstrap 95% CI for Spearman GTC
        bootstrap_spearman = []
        for _ in range(n_bootstrap):
            resample = np.random.choice(spearman_corrs, size=len(spearman_corrs), replace=True)
            bootstrap_spearman.append(np.mean(resample))
        spearman_ci_lower = np.percentile(bootstrap_spearman, 2.5)
        spearman_ci_upper = np.percentile(bootstrap_spearman, 97.5)
        spearman_mean = np.mean(spearman_corrs)
        
        # Energy-in-mask stats
        energy_mean = np.mean(energy_vals)
        energy_std = np.std(energy_vals)
        
        results[model_name] = {
            'pearson_mean': mean_val,
            'pearson_ci_lower': ci_lower,
            'pearson_ci_upper': ci_upper,
            'spearman_mean': spearman_mean,
            'spearman_ci_lower': spearman_ci_lower,
            'spearman_ci_upper': spearman_ci_upper,
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'n_samples': len(corrs),
            'correlations': corrs
        }
        
        short_name = model_name.replace('BLiSS_', '')
        print(f"\n  【{short_name}】 (n={len(corrs)})")
        print(f"    Pearson GTC:  {mean_val:.4f}  [95% CI: {ci_lower:.4f} - {ci_upper:.4f}]")
        print(f"    Spearman GTC: {spearman_mean:.4f}  [95% CI: {spearman_ci_lower:.4f} - {spearman_ci_upper:.4f}]")
        print(f"    Energy-in-mask: {energy_mean:.4f} ± {energy_std:.4f}")
    
    # Paired t-tests between models
    print("\n" + "-"*70)
    print("  【Paired t-test Results (Pearson GTC)】")
    
    if len(model_names) >= 2:
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                name1, name2 = model_names[i], model_names[j]
                if name1 not in results or name2 not in results:
                    continue
                    
                corrs1 = results[name1]['correlations']
                corrs2 = results[name2]['correlations']
                
                # Ensure same length for paired test
                min_len = min(len(corrs1), len(corrs2))
                if min_len < 2:
                    continue
                
                t_stat, p_value = stats.ttest_rel(corrs1[:min_len], corrs2[:min_len])
                
                short1 = name1.replace('BLiSS_', '')
                short2 = name2.replace('BLiSS_', '')
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"    {short1} vs {short2}: t={t_stat:.3f}, p={p_value:.4f} {sig}")
    
    # Save results to JSON
    json_results = {}
    for model_name, data in results.items():
        json_results[model_name] = {k: v for k, v in data.items() if k != 'correlations'}
    
    json_path = os.path.join(save_dir, 'statistical_analysis.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ 详细结果保存至: {json_path}")
    
    return results


def main():
    """主函数"""
    print("\n" + "="*70)
    print("BLiSS Enhanced Visualization Comparison")
    print("="*70)
    
    # 构建参数
    import sys
    old_argv = sys.argv
    sys.argv = ['', '--dataset=BLiSS']
    args = build_args()
    sys.argv = old_argv
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n使用设备: {args.device}")
    
    # 加载数据集
    print("\n加载BLiSS数据集...")
    test_set = BLiSSDataset(mode='test', args=args)
    print(f"  ✓ 共 {len(test_set)} 个测试样本")
    
    # 查找所有模型
    bliss_log_dir = os.path.join(os.path.dirname(__file__), 'logs', 'BLiss')
    model_dirs = {}
    
    for name in os.listdir(bliss_log_dir):
        model_path = os.path.join(bliss_log_dir, name)
        if os.path.isdir(model_path):
            model_dirs[name] = model_path
    
    print(f"\n发现 {len(model_dirs)} 个模型:")
    for name in model_dirs:
        print(f"  • {name}")
    
    # 分析每个模型
    models_data = {}
    num_samples_to_analyze = min(1000, len(test_set))
    
    for model_name, model_path in model_dirs.items():
        print(f"\n{'='*50}")
        print(f"分析模型: {model_name}")
        print("="*50)
        
        try:
            # 更新model_dir
            args.model_dir = model_path
            model = load_model(model_path, args)
            print(f"  ✓ 模型加载成功")
            
            samples_results = []
            for idx in tqdm(range(num_samples_to_analyze), desc="分析样本"):
                sample = test_set[idx]
                
                # 检查是否有有效的对齐mask
                if sample[10].dim() < 2:  # video_to_text_mask
                    continue
                
                result = compute_causal_matrix(model, sample, args)
                samples_results.append(result)
            
            models_data[model_name] = {
                'samples': samples_results,
                'avg_correlation': np.nanmean([s['correlation'] for s in samples_results]),
                'avg_spearman': np.nanmean([s['spearman_correlation'] for s in samples_results]),
                'avg_energy_in_mask': np.nanmean([s['energy_in_mask'] for s in samples_results]),
                'avg_var_ratio': np.mean([s['var_ratio'] for s in samples_results]),
                'correlations': [s['correlation'] for s in samples_results if not np.isnan(s['correlation'])]
            }
            
            print(f"  平均GT相关性 (Pearson): {models_data[model_name]['avg_correlation']:.4f}")
            print(f"  平均GT相关性 (Spearman): {models_data[model_name]['avg_spearman']:.4f}")
            print(f"  平均Energy-in-mask: {models_data[model_name]['avg_energy_in_mask']:.4f}")
            print(f"  平均Var Ratio: {models_data[model_name]['avg_var_ratio']:.4f}")
            
        except Exception as e:
            print(f"  ⚠️ 分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 创建保存目录
    save_dir = './interpretability_results/bliss_comparison'
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成可视化
    if models_data:
        print("\n" + "="*70)
        print("生成可视化...")
        print("="*70)
        
        visualize_alignment_comparison(models_data, save_dir)
        visualize_correlation_distribution(models_data, save_dir)
        visualize_intervention_effects(models_data, save_dir)
        generate_summary_figure(models_data, save_dir)
        
        # 统计显著性分析
        compute_statistical_significance(models_data, save_dir)
        
        print("\n" + "="*70)
        print(f"所有可视化已保存至: {save_dir}")
        print("="*70)
        
        # 打印文件列表
        print("\n生成的文件:")
        for f in os.listdir(save_dir):
            if f.endswith('.png') or f.endswith('.json'):
                print(f"  • {f}")
    else:
        print("\n⚠️ 没有成功分析的模型")


if __name__ == '__main__':
    main()
