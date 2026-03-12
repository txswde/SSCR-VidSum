#!/usr/bin/env python3
"""
SumMe 无监督 vs 无监督+因果 可解释性对比分析

比较两组模型的内部行为：
1. 因果效应矩阵的结构 (是否有对角线模式)
2. GT相关性 (对齐区域的效应是否更强)
3. 注意力集中度 (Row/Col 方差比)
4. 峰值位置相关性

这是证明"因果对齐"价值的关键分析 - 即使F1相同，因果模型应该有更好的可解释性。
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.utils.rnn import pad_sequence
from scipy import stats
from pathlib import Path
import yaml
import json
from typing import Dict, List, Tuple

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class SumMeInterpretabilityAnalyzer:
    """SumMe可解释性对比分析器"""
    
    def __init__(self, model, args, model_name: str):
        self.model = model
        self.args = args
        self.device = args.device
        self.model_name = model_name
        self.model.eval()
        
    def compute_causal_effect_matrix(self, video, text, mask_video, mask_text,
                                     video_label, text_label,
                                     video_to_text_mask_list, text_to_video_mask_list) -> Dict:
        """计算因果效应矩阵"""
        batch_size, num_sentences, _ = text.shape
        num_frames = video.shape[1]
        
        # 1. 计算原始预测
        with torch.no_grad():
            output = self.model(
                video=video, text=text,
                mask_video=mask_video, mask_text=mask_text,
                video_label=video_label, text_label=text_label,
                video_to_text_mask_list=video_to_text_mask_list,
                text_to_video_mask_list=text_to_video_mask_list
            )
            if isinstance(output, dict):
                orig_video_score = output['video_pred_cls']
            else:
                orig_video_score = output[0] if isinstance(output, tuple) else output
            orig_probs = torch.sigmoid(orig_video_score)
        
        # 2. 获取期望对齐模式 (GT)
        expected_alignment = video_to_text_mask_list[0].cpu().numpy().T  # [num_sentences, num_frames]
        
        # 3. 计算因果影响矩阵
        causal_impact_matrix = torch.zeros(batch_size, num_sentences, num_frames).to(self.device)
        
        for sent_idx in range(num_sentences):
            cf_mask_text = mask_text.clone()
            is_valid = cf_mask_text[:, sent_idx] == 1
            if not is_valid.any():
                continue
            cf_mask_text[:, sent_idx] = 0
            
            with torch.no_grad():
                output = self.model(
                    video=video, text=text,
                    mask_video=mask_video, mask_text=cf_mask_text,
                    video_label=video_label, text_label=text_label,
                    video_to_text_mask_list=video_to_text_mask_list,
                    text_to_video_mask_list=text_to_video_mask_list
                )
                if isinstance(output, dict):
                    cf_video_score = output['video_pred_cls']
                else:
                    cf_video_score = output[0] if isinstance(output, tuple) else output
                cf_probs = torch.sigmoid(cf_video_score)
            
            effect = orig_probs - cf_probs
            causal_impact_matrix[:, sent_idx, :] = effect
        
        return {
            'causal_impact': causal_impact_matrix,
            'expected_alignment': expected_alignment,
            'orig_probs': orig_probs
        }
    
    def compute_diagnostics(self, causal_matrix: np.ndarray, expected_alignment: np.ndarray) -> Dict:
        """计算诊断指标"""
        # 确保维度匹配
        n_sent, n_frame = causal_matrix.shape
        exp_n_sent, exp_n_frame = expected_alignment.shape
        
        min_sent = min(n_sent, exp_n_sent)
        min_frame = min(n_frame, exp_n_frame)
        
        causal_sub = causal_matrix[:min_sent, :min_frame]
        expected_sub = expected_alignment[:min_sent, :min_frame]
        
        # 1. GT相关性 (Alignment Ratio)
        aligned_effects = causal_sub[expected_sub > 0]
        non_aligned_effects = causal_sub[expected_sub == 0]
        
        if len(aligned_effects) > 0 and len(non_aligned_effects) > 0:
            gt_relevance = np.mean(np.abs(aligned_effects)) / (np.mean(np.abs(non_aligned_effects)) + 1e-8)
        else:
            gt_relevance = 0
        
        # 2. 行方差 (句子特异性)
        row_variances = np.var(causal_sub, axis=1)
        avg_row_variance = np.mean(row_variances)
        
        # 3. 列方差 (帧特异性)
        col_variances = np.var(causal_sub, axis=0)
        avg_col_variance = np.mean(col_variances)
        
        # Row/Col 方差比
        row_col_ratio = avg_row_variance / (avg_col_variance + 1e-8)
        
        # 4. 峰值位置相关性
        peak_positions = np.argmax(np.abs(causal_sub), axis=1)
        expected_positions = np.array([
            np.mean(np.where(expected_sub[i] > 0)[0]) 
            if np.any(expected_sub[i] > 0) else i * min_frame / min_sent 
            for i in range(min_sent)
        ])
        
        if len(peak_positions) > 2:
            peak_corr, peak_p = stats.spearmanr(peak_positions, expected_positions)
        else:
            peak_corr, peak_p = 0, 1
        
        # 5. 对角线得分 (简化版)
        diag_score = 0
        for i in range(min(min_sent, min_frame)):
            row_idx = int(i * min_sent / min(min_sent, min_frame))
            col_idx = int(i * min_frame / min(min_sent, min_frame))
            if row_idx < min_sent and col_idx < min_frame:
                diag_score += np.abs(causal_sub[row_idx, col_idx])
        diag_score /= min(min_sent, min_frame)
        
        # 全局平均效应
        global_avg = np.mean(np.abs(causal_sub))
        relative_diag = diag_score / (global_avg + 1e-8)
        
        return {
            'gt_relevance': float(gt_relevance),
            'row_variance': float(avg_row_variance),
            'col_variance': float(avg_col_variance),
            'row_col_ratio': float(row_col_ratio),
            'peak_correlation': float(peak_corr),
            'peak_p_value': float(peak_p),
            'diagonal_score': float(diag_score),
            'relative_diagonal': float(relative_diag),
            'global_avg_effect': float(global_avg)
        }


def load_model_and_data(model_dir: str, args):
    """加载模型和数据"""
    from models import Model_VideoSumm
    from datasets import VideoSummDataset, my_collate_fn
    
    # 加载配置
    args_path = Path(model_dir) / 'args.yml'
    if args_path.exists():
        with open(args_path, 'r') as f:
            saved_args = yaml.safe_load(f)
        # 更新关键参数
        for key in ['disable_alignment_mask', 'enable_causal_alignment']:
            if key in saved_args:
                setattr(args, key, saved_args[key])
    
    # 加载数据 - 使用split 0
    split_path = f'{args.data_root}/{args.dataset}/splits.yml'
    with open(split_path, 'r') as f:
        splits = yaml.safe_load(f)
    
    val_keys = splits[0]['test_keys']
    val_dataset = VideoSummDataset(val_keys, args=args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=my_collate_fn
    )
    
    # 加载模型
    checkpoint_path = Path(model_dir) / 'checkpoint' / 'model_best_split0.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = Model_VideoSumm(args=args)
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    return model, val_loader


def run_comparison_analysis(args, output_dir: Path, num_samples: int = 5):
    """运行对比分析"""
    from datasets import my_collate_fn
    
    print("=" * 60)
    print("SumMe 可解释性对比分析")
    print("无监督 (Unsupervised) vs 无监督+因果 (Unsupervised+Causal)")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义要比较的模型
    models_config = {
        'Unsupervised': 'logs/SumMe_unsupervised',
        'Unsupervised+Causal': 'logs/SumMe_unsupervised_causal'
    }
    
    # 存储所有诊断结果
    all_results = {name: [] for name in models_config}
    causal_matrices = {name: [] for name in models_config}
    
    # 首先加载数据 (只需要加载一次)
    data_loader = None
    
    for model_name, model_dir in models_config.items():
        print(f"\n{'='*40}")
        print(f"分析模型: {model_name}")
        print(f"目录: {model_dir}")
        print("=" * 40)
        
        try:
            model, val_loader = load_model_and_data(model_dir, args)
            if data_loader is None:
                data_loader = val_loader
            print(f"✓ 模型加载成功")
        except Exception as e:
            print(f"✗ 加载失败: {e}")
            continue
        
        analyzer = SumMeInterpretabilityAnalyzer(model, args, model_name)
        
        for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break
            
            print(f"  处理样本 {i+1}/{num_samples}...")
            
            video_list, text_list, mask_video_list, mask_text_list, \
            video_cls_label_list, video_loc_label_list, video_ctr_label_list, \
            text_cls_label_list, text_loc_label_list, text_ctr_label_list, \
            _, _, _, _, _, _, video_to_text_mask_list, text_to_video_mask_list, _, _ = batch
            
            video = pad_sequence(video_list, batch_first=True).to(args.device)
            text = pad_sequence(text_list, batch_first=True).to(args.device)
            mask_video = pad_sequence(mask_video_list, batch_first=True).to(args.device)
            mask_text = pad_sequence(mask_text_list, batch_first=True).to(args.device)
            video_label = pad_sequence(video_cls_label_list, batch_first=True).to(args.device)
            text_label = pad_sequence(text_cls_label_list, batch_first=True).to(args.device)
            
            for j in range(len(video_to_text_mask_list)):
                video_to_text_mask_list[j] = video_to_text_mask_list[j].to(args.device)
                text_to_video_mask_list[j] = text_to_video_mask_list[j].to(args.device)
            
            # 计算因果效应
            results = analyzer.compute_causal_effect_matrix(
                video=video, text=text,
                mask_video=mask_video, mask_text=mask_text,
                video_label=video_label, text_label=text_label,
                video_to_text_mask_list=video_to_text_mask_list,
                text_to_video_mask_list=text_to_video_mask_list
            )
            
            causal_matrix = results['causal_impact'][0].cpu().numpy()
            expected_alignment = results['expected_alignment']
            
            # 过滤无效行列
            rows_keep = ~np.all(causal_matrix == 0, axis=1)
            cols_keep = ~np.all(causal_matrix == 0, axis=0)
            
            if np.any(rows_keep) and np.any(cols_keep):
                causal_vis = causal_matrix[rows_keep][:, cols_keep]
                diag = analyzer.compute_diagnostics(causal_vis, expected_alignment)
                all_results[model_name].append(diag)
                causal_matrices[model_name].append(causal_vis)
        
        # 清理
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 汇总比较
    print("\n" + "=" * 60)
    print("可解释性指标对比")
    print("=" * 60)
    
    summary = {}
    for model_name, results_list in all_results.items():
        if not results_list:
            continue
        
        summary[model_name] = {
            'gt_relevance': {
                'mean': np.mean([r['gt_relevance'] for r in results_list]),
                'std': np.std([r['gt_relevance'] for r in results_list])
            },
            'row_col_ratio': {
                'mean': np.mean([r['row_col_ratio'] for r in results_list]),
                'std': np.std([r['row_col_ratio'] for r in results_list])
            },
            'peak_correlation': {
                'mean': np.mean([r['peak_correlation'] for r in results_list]),
                'std': np.std([r['peak_correlation'] for r in results_list])
            },
            'relative_diagonal': {
                'mean': np.mean([r['relative_diagonal'] for r in results_list]),
                'std': np.std([r['relative_diagonal'] for r in results_list])
            },
            'n_samples': len(results_list)
        }
        
        print(f"\n{model_name}:")
        print(f"  GT相关性 (越高越好): {summary[model_name]['gt_relevance']['mean']:.4f} ± {summary[model_name]['gt_relevance']['std']:.4f}")
        print(f"  Row/Col方差比: {summary[model_name]['row_col_ratio']['mean']:.4f} ± {summary[model_name]['row_col_ratio']['std']:.4f}")
        print(f"  峰值相关性: {summary[model_name]['peak_correlation']['mean']:.4f} ± {summary[model_name]['peak_correlation']['std']:.4f}")
        print(f"  相对对角线得分: {summary[model_name]['relative_diagonal']['mean']:.4f} ± {summary[model_name]['relative_diagonal']['std']:.4f}")
    
    # 创建可视化
    create_comparison_visualizations(all_results, causal_matrices, summary, output_dir)
    
    # 生成报告
    generate_interpretability_report(all_results, summary, output_dir)
    
    # 保存JSON
    json_path = output_dir / 'interpretability_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 分析完成! 结果保存至: {output_dir}")
    return summary


def create_comparison_visualizations(all_results: Dict, causal_matrices: Dict, summary: Dict, output_dir: Path):
    """创建对比可视化"""
    
    # 1. 指标对比柱状图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    metrics = ['gt_relevance', 'row_col_ratio', 'peak_correlation', 'relative_diagonal']
    metric_labels = ['GT Relevance\n(Higher=Better)', 'Row/Col Variance Ratio', 
                    'Peak Position Correlation', 'Relative Diagonal Score']
    colors = ['#e74c3c', '#2ecc71']
    
    for ax, metric, label in zip(axes.flat, metrics, metric_labels):
        names = list(summary.keys())
        means = [summary[n][metric]['mean'] for n in names]
        stds = [summary[n][metric]['std'] for n in names]
        
        x = np.arange(len(names))
        bars = ax.bar(x, means, yerr=stds, capsize=10, color=colors[:len(names)], alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, mean in zip(bars, means):
            ax.annotate(f'{mean:.3f}', xy=(bar.get_x() + bar.get_width()/2, mean),
                       xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'interpretability_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存: interpretability_metrics_comparison.png")
    
    # 2. 因果矩阵对比热力图 (取第一个样本)
    if all(len(causal_matrices[k]) > 0 for k in causal_matrices):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for ax, (model_name, matrices) in zip(axes, causal_matrices.items()):
            matrix = matrices[0]  # 第一个样本
            # 标准化
            matrix_norm = (matrix - np.mean(matrix)) / (np.std(matrix) + 1e-9)
            sns.heatmap(matrix_norm, cmap="RdBu_r", center=0, vmin=-2, vmax=2, ax=ax, cbar=True)
            ax.set_title(f'{model_name}\nCausal Effect Matrix (Sample 1)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Video Frames', fontsize=12)
            ax.set_ylabel('Text Sentences', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'causal_matrix_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存: causal_matrix_comparison.png")
    
    # 3. 样本级对比箱线图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    metric_pairs = [('gt_relevance', 'GT Relevance'), ('peak_correlation', 'Peak Correlation')]
    
    for ax, (metric, label) in zip(axes, metric_pairs):
        data = []
        labels = []
        for model_name, results in all_results.items():
            if results:
                data.append([r[metric] for r in results])
                labels.append(model_name)
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label} Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_level_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存: sample_level_comparison.png")


def generate_interpretability_report(all_results: Dict, summary: Dict, output_dir: Path):
    """生成可解释性报告"""
    
    names = list(summary.keys())
    
    # 计算统计显著性
    if len(names) == 2 and all(len(all_results[n]) > 0 for n in names):
        gt_rel_1 = [r['gt_relevance'] for r in all_results[names[0]]]
        gt_rel_2 = [r['gt_relevance'] for r in all_results[names[1]]]
        
        if len(gt_rel_1) > 2 and len(gt_rel_2) > 2:
            t_stat, p_val = stats.ttest_ind(gt_rel_1, gt_rel_2)
        else:
            t_stat, p_val = 0, 1
    else:
        t_stat, p_val = 0, 1
    
    report = f"""# SumMe 可解释性对比分析报告

## 📊 核心发现

虽然两个模型在F1-score上几乎相同，但**可解释性指标**可以揭示它们内部机制的差异。

## 🔬 可解释性指标对比

| 指标 | {names[0] if len(names) > 0 else 'Model 1'} | {names[1] if len(names) > 1 else 'Model 2'} | 说明 |
|------|------------|--------------|------|
"""
    
    if len(names) >= 2:
        for metric, label, interp in [
            ('gt_relevance', 'GT相关性', '对齐区域效应/非对齐区域效应'),
            ('row_col_ratio', 'Row/Col方差比', '>1表示水平条纹(好), <1表示垂直条纹(差)'),
            ('peak_correlation', '峰值相关性', '因果峰值位置与期望位置的相关性'),
            ('relative_diagonal', '相对对角线', '对角线效应/全局平均效应')
        ]:
            v1 = summary[names[0]][metric]['mean']
            v2 = summary[names[1]][metric]['mean']
            report += f"| {label} | {v1:.4f} ± {summary[names[0]][metric]['std']:.4f} | {v2:.4f} ± {summary[names[1]][metric]['std']:.4f} | {interp} |\n"
    
    report += f"""
## 📈 统计检验

- **GT相关性 t检验**: t = {t_stat:.4f}, p = {p_val:.4f}
- **显著性 (α=0.05)**: {'✅ 显著' if p_val < 0.05 else '❌ 不显著'}

## 🔍 可视化

### 可解释性指标对比
![Metrics Comparison](interpretability_metrics_comparison.png)

### 因果矩阵热力图
![Causal Matrix](causal_matrix_comparison.png)

*红色表示正向因果效应(移除句子导致帧得分下降), 蓝色表示负向效应*

### 样本级分布对比
![Sample Distribution](sample_level_comparison.png)

## 🎯 结论
"""
    
    if len(names) >= 2:
        gt1 = summary[names[0]]['gt_relevance']['mean']
        gt2 = summary[names[1]]['gt_relevance']['mean']
        
        if gt2 > gt1 * 1.1:  # 因果模型GT相关性更高10%以上
            report += f"""
> ✅ **因果对齐有效**: {names[1]} 的 GT相关性 ({gt2:.4f}) 高于 {names[0]} ({gt1:.4f})。
> 
> 这意味着因果模型的注意力更倾向于"正确"的区域（Ground Truth定义的对齐区域），
> 即使两者的F1分数相同，因果模型的决策过程更加**可解释**和**符合人类直觉**。
"""
        elif gt1 > gt2 * 1.1:
            report += f"""
> ⚠️ **无监督模型更优**: {names[0]} 的 GT相关性 ({gt1:.4f}) 反而高于 {names[1]} ({gt2:.4f})。
> 
> 这可能说明因果约束过强，或者需要调整 `lambda_causal_alignment` 参数。
"""
        else:
            report += f"""
> ➡️ **两者持平**: GT相关性差异不大 ({gt1:.4f} vs {gt2:.4f})。
> 
> 在SumMe数据集上，因果约束未能明显改善可解释性，可能原因：
> 1. SumMe数据集较小，信号不足
> 2. 因果损失权重需要调整
> 3. 可能需要更多训练轮数才能显现差异
"""
    
    report += f"""
## 💡 建议

1. **如果因果模型GT相关性更高**: 在论文中强调"虽然F1持平，但模型的可解释性更好"
2. **如果两者持平**: 考虑在BLiSS等更大数据集上验证因果效果
3. **进一步分析**: 可视化具体视频的注意力图，找到典型的"因果更好"的案例

---

*生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    report_path = output_dir / 'interpretability_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"  保存: interpretability_report.md")


if __name__ == '__main__':
    import argparse
    import sys
    
    # 先解析自定义参数
    custom_parser = argparse.ArgumentParser(description='SumMe可解释性对比分析', add_help=False)
    custom_parser.add_argument('--output-dir', type=str, default='summe_interpretability_comparison',
                               help='输出目录')
    custom_parser.add_argument('--num-samples', type=int, default=5,
                               help='分析的样本数量')
    
    custom_args, remaining = custom_parser.parse_known_args()
    
    # 过滤掉自定义参数后再获取config参数，并添加dataset参数
    sys.argv = [sys.argv[0], '--dataset', 'SumMe'] + remaining
    
    from config import get_arguments
    args = get_arguments()
    
    output_dir = Path(custom_args.output_dir)
    
    summary = run_comparison_analysis(args, output_dir, num_samples=custom_args.num_samples)
