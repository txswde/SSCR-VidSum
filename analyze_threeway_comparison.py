#!/usr/bin/env python3
"""
三组实验对比分析脚本

比较以下三个实验:
1. Supervised Baseline (监督基线) - 使用GT对齐标签
2. Unsupervised (无监督) - 禁用对齐标签
3. Unsupervised + Causal (无监督+因果) - 核心创新

分析内容:
1. F1-score对比
2. 统计显著性检验 (t-test, ANOVA, Tukey HSD)
3. 效应量计算
4. 训练参数对比
5. 可视化对比图
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json
import yaml
import argparse

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def extract_fscores_from_log(log_path: Path, num_splits: int = 5) -> Tuple[List[float], float, float]:
    """
    从训练日志中提取每个split的最佳F-score
    
    Args:
        log_path: 日志文件路径
        num_splits: split数量 (默认5)
        
    Returns:
        (最佳F-scores列表, 平均F1, 标准差)
    """
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 方法1: 从 F1_results 行提取
    f1_results_pattern = r"F1_results: \{([^}]+)\}"
    all_matches = re.findall(f1_results_pattern, content)
    
    best_scores = []
    
    if all_matches:
        f1_str = all_matches[-1]
        split_scores = re.findall(r"'split(\d+)':\s*([\d.]+)", f1_str)
        for split_id, score in split_scores:
            best_scores.append(float(score))
    
    # 方法2: 从最终摘要行提取平均F1
    final_f1_pattern = r"F1-score:\s*([\d.]+)"
    all_final_matches = re.findall(final_f1_pattern, content)
    
    if all_final_matches:
        mean_f1 = float(all_final_matches[-1])
    elif best_scores:
        mean_f1 = np.mean(best_scores)
    else:
        mean_f1 = 0.0
    
    std_f1 = np.std(best_scores) if best_scores else 0.0
    
    return best_scores, mean_f1, std_f1


def load_args(args_path: Path) -> Dict:
    """加载训练配置"""
    try:
        with open(args_path, 'r') as f:
            args = yaml.safe_load(f)
        return args
    except Exception as e:
        print(f"Warning: Could not load config from {args_path}: {e}")
        return {}


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """计算Cohen's d效应量"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


def interpret_cohens_d(d: float) -> str:
    """解释Cohen's d效应量"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"


def pairwise_comparisons(data: Dict[str, np.ndarray]) -> Dict:
    """执行成对t检验"""
    results = {}
    names = list(data.keys())
    
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            name1, name2 = names[i], names[j]
            arr1, arr2 = data[name1], data[name2]
            
            # 配对t检验
            t_stat, p_value = stats.ttest_rel(arr1, arr2)
            
            # Cohen's d
            d = cohens_d(arr1, arr2)
            
            results[f"{name1} vs {name2}"] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohens_d': float(d),
                'interpretation': interpret_cohens_d(d),
                'significant': bool(p_value < 0.05),
                'mean_diff': float(np.mean(arr1) - np.mean(arr2))
            }
    
    return results


def one_way_anova(data: Dict[str, np.ndarray]) -> Dict:
    """执行单因素方差分析"""
    groups = list(data.values())
    f_stat, p_value = stats.f_oneway(*groups)
    
    return {
        'f_statistic': float(f_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
        'num_groups': len(groups)
    }


def create_visualizations(exp_data: Dict, output_dir: Path):
    """创建可视化图表"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备数据
    names = list(exp_data.keys())
    all_scores = [exp_data[name]['split_scores'] for name in names]
    means = [exp_data[name]['mean_f1'] for name in names]
    stds = [exp_data[name]['std_f1'] for name in names]
    
    # 设置颜色
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # 蓝、红、绿
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 图1: 箱线图对比
    ax1 = axes[0]
    bp = ax1.boxplot(all_scores, labels=names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_ylabel('F-score', fontsize=12)
    ax1.set_title('F-score Distribution by Method', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=15)
    
    # 图2: 柱状图 (带误差棒)
    ax2 = axes[1]
    x = np.arange(len(names))
    bars = ax2.bar(x, means, yerr=stds, capsize=10, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.set_ylabel('F-score', fontsize=12)
    ax2.set_title('Mean F-score Comparison (±std)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, mean, std in zip(bars, means, stds):
        ax2.annotate(f'{mean:.4f}', xy=(bar.get_x() + bar.get_width()/2, mean + std + 0.005),
                    ha='center', fontsize=10, fontweight='bold')
    
    # 图3: 各Split配对对比
    ax3 = axes[2]
    splits = range(len(all_scores[0]))
    for idx, (name, scores) in enumerate(zip(names, all_scores)):
        ax3.plot(splits, scores, 'o-', label=name, color=colors[idx], 
                linewidth=2, markersize=8, alpha=0.8)
    ax3.set_xlabel('Split', fontsize=12)
    ax3.set_ylabel('F-score', fontsize=12)
    ax3.set_title('Per-Split Performance Comparison', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(list(splits))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threeway_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_dir / 'threeway_comparison.png'}")
    
    # 图4: 热力图 - 成对差异
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 创建差异矩阵
    diff_matrix = np.zeros((len(names), len(names)))
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            diff_matrix[i, j] = means[i] - means[j]
    
    im = ax.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.05, vmax=0.05)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticklabels(names)
    
    # 添加数值注释
    for i in range(len(names)):
        for j in range(len(names)):
            text = ax.text(j, i, f'{diff_matrix[i, j]:.4f}',
                          ha='center', va='center', color='white' if abs(diff_matrix[i,j]) > 0.02 else 'black',
                          fontweight='bold')
    
    ax.set_title('Pairwise Mean F-score Difference\n(Row - Column)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Difference')
    plt.tight_layout()
    plt.savefig(output_dir / 'pairwise_difference_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to {output_dir / 'pairwise_difference_heatmap.png'}")


def generate_report(exp_data: Dict, pairwise_results: Dict, anova_results: Dict, output_path: Path):
    """生成Markdown格式的统计报告"""
    
    names = list(exp_data.keys())
    
    report = f"""# 三组实验对比分析报告

## 📊 实验概览

| 实验 | 平均F1 | 标准差 | 各Split得分 |
|------|--------|--------|-------------|
"""
    
    for name in names:
        data = exp_data[name]
        scores_str = ', '.join([f'{s:.4f}' for s in data['split_scores']])
        report += f"| {name} | **{data['mean_f1']:.4f}** | {data['std_f1']:.4f} | {scores_str} |\n"
    
    report += f"""
## 🔬 配置差异

| 参数 | Supervised | Unsupervised | Unsupervised+Causal |
|------|------------|--------------|---------------------|
"""
    
    key_params = ['disable_alignment_mask', 'enable_causal_alignment', 'seed', 'max_epoch']
    for param in key_params:
        values = [str(exp_data[name].get('args', {}).get(param, 'N/A')) for name in names]
        report += f"| {param} | {' | '.join(values)} |\n"
    
    report += f"""
## 📈 统计分析

### 1. 单因素方差分析 (One-Way ANOVA)

- **F统计量**: {anova_results['f_statistic']:.4f}
- **p值**: {anova_results['p_value']:.4f}
- **显著性 (α=0.05)**: {'✅ 显著' if anova_results['significant'] else '❌ 不显著'}

> {'组间存在显著差异，需要进行事后检验' if anova_results['significant'] else '组间无显著差异'}

### 2. 成对t检验 (Pairwise Comparisons)

| 对比 | t统计量 | p值 | Cohen's d | 效应解释 | 平均差值 | 结论 |
|------|---------|-----|-----------|----------|----------|------|
"""
    
    for comparison, results in pairwise_results.items():
        sig_mark = '✅' if results['significant'] else '❌'
        report += f"| {comparison} | {results['t_statistic']:.4f} | {results['p_value']:.4f} | {results['cohens_d']:.4f} | {results['interpretation']} | {results['mean_diff']:+.4f} | {sig_mark} |\n"
    
    report += f"""
### 3. 效应量解释

| 效应大小 | Cohen's d范围 | 含义 |
|----------|---------------|------|
| Negligible | |d| < 0.2 | 可忽略的差异 |
| Small | 0.2 ≤ |d| < 0.5 | 小效应 |
| Medium | 0.5 ≤ |d| < 0.8 | 中等效应 |
| Large | |d| ≥ 0.8 | 大效应 |

## 📊 可视化

### F-score对比图
![Threeway Comparison](threeway_comparison.png)

*左: 各方法F-score分布箱线图; 中: 平均F-score柱状图(带标准差); 右: 各Split配对对比线图*

### 成对差异热力图
![Pairwise Difference](pairwise_difference_heatmap.png)

*颜色表示行减去列的平均F-score差异*

## 🎯 结论与见解

### 关键发现
"""
    
    # 按平均F1排序
    sorted_names = sorted(names, key=lambda x: exp_data[x]['mean_f1'], reverse=True)
    report += f"\n1. **最佳表现**: {sorted_names[0]} (F1 = {exp_data[sorted_names[0]]['mean_f1']:.4f})\n"
    report += f"2. **次优表现**: {sorted_names[1]} (F1 = {exp_data[sorted_names[1]]['mean_f1']:.4f})\n"
    report += f"3. **第三表现**: {sorted_names[2]} (F1 = {exp_data[sorted_names[2]]['mean_f1']:.4f})\n"
    
    # 比较监督与无监督
    supervised_f1 = exp_data['Supervised']['mean_f1']
    unsupervised_f1 = exp_data['Unsupervised']['mean_f1']
    causal_f1 = exp_data['Unsupervised+Causal']['mean_f1']
    
    report += f"""
### 监督 vs 无监督分析

- 监督基线: F1 = {supervised_f1:.4f}
- 无监督版本: F1 = {unsupervised_f1:.4f} (相差 {unsupervised_f1 - supervised_f1:+.4f})
- 无监督+因果: F1 = {causal_f1:.4f} (相差 {causal_f1 - supervised_f1:+.4f})

### 因果对齐效果

- 无因果对齐: F1 = {unsupervised_f1:.4f}
- 有因果对齐: F1 = {causal_f1:.4f}
- 差异: {causal_f1 - unsupervised_f1:+.4f}

### 重要观察

"""
    
    # 添加基于结果的观察
    if supervised_f1 > unsupervised_f1 and supervised_f1 > causal_f1:
        report += "> ⚠️ **监督学习仍然最优**: 这表明GT对齐标签提供了有价值的监督信号。\n\n"
    elif causal_f1 > unsupervised_f1:
        report += "> ✅ **因果对齐有正向效果**: 无监督+因果版本优于纯无监督版本。\n\n"
    else:
        report += "> ⚠️ **因果对齐效果尚不明显**: 可能需要调整超参数或增加训练轮数。\n\n"
    
    if anova_results['p_value'] > 0.05:
        report += "> 📊 **统计上无显著差异**: ANOVA结果表明三组方法在当前样本量下无统计显著差异，可能需要更多实验重复或更大的数据集来验证效果。\n\n"
    
    report += f"""
## 🔧 建议

1. **增加样本量**: 当前每组仅5个splits，统计功效较低，建议增加实验重复次数
2. **超参数调优**: 考虑调整`lambda_causal_alignment`等因果相关超参数
3. **可视化注意力**: 使用`enhanced_causal_analysis.py`可视化因果注意力矩阵，分析模型学到的对齐模式
4. **跨数据集验证**: 在TVSum等其他数据集上重复实验以验证结论的泛化性

---

*生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
*分析脚本: analyze_threeway_comparison.py*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='三组实验对比分析')
    parser.add_argument('--supervised-log', type=str, 
                       default='logs/SumMe_supervised_baseline',
                       help='监督基线实验目录')
    parser.add_argument('--unsupervised-log', type=str,
                       default='logs/SumMe_unsupervised',  
                       help='无监督实验目录')
    parser.add_argument('--causal-log', type=str,
                       default='logs/SumMe_unsupervised_causal',  
                       help='无监督+因果实验目录')
    parser.add_argument('--output-dir', type=str, default='summe_threeway_analysis',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SumMe 三组实验对比分析")
    print("=" * 60)
    
    # 定义实验
    experiments = {
        'Supervised': args.supervised_log,
        'Unsupervised': args.unsupervised_log,
        'Unsupervised+Causal': args.causal_log
    }
    
    # 提取数据
    exp_data = {}
    for name, log_dir in experiments.items():
        log_path = Path(log_dir) / 'log.txt'
        args_path = Path(log_dir) / 'args.yml'
        
        print(f"\n提取 {name} ({log_path})...")
        
        if not log_path.exists():
            print(f"  警告: 日志文件不存在!")
            continue
            
        split_scores, mean_f1, std_f1 = extract_fscores_from_log(log_path)
        config = load_args(args_path)
        
        exp_data[name] = {
            'split_scores': split_scores,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'args': config
        }
        
        print(f"  平均F1: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"  各Split: {[f'{s:.4f}' for s in split_scores]}")
    
    if len(exp_data) < 3:
        print(f"\n警告: 只找到 {len(exp_data)} 个实验，需要3个才能进行完整对比分析!")
        return
    
    # 准备数据用于统计分析
    data_arrays = {name: np.array(exp_data[name]['split_scores']) for name in exp_data}
    
    # 执行统计分析
    print("\n执行统计分析...")
    
    # 单因素方差分析
    anova_results = one_way_anova(data_arrays)
    print(f"  ANOVA: F={anova_results['f_statistic']:.4f}, p={anova_results['p_value']:.4f}")
    
    # 成对t检验
    pairwise_results = pairwise_comparisons(data_arrays)
    for comparison, results in pairwise_results.items():
        print(f"  {comparison}: t={results['t_statistic']:.4f}, p={results['p_value']:.4f}, d={results['cohens_d']:.4f}")
    
    # 保存统计结果JSON
    all_results = {
        'experiments': {name: {k: v for k, v in data.items() if k != 'args'} for name, data in exp_data.items()},
        'anova': anova_results,
        'pairwise_comparisons': pairwise_results
    }
    
    json_path = output_dir / 'threeway_analysis_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON results saved to {json_path}")
    
    # 创建可视化
    print("\n生成可视化...")
    create_visualizations(exp_data, output_dir)
    
    # 生成报告
    print("\n生成统计报告...")
    report_path = output_dir / 'threeway_analysis_report.md'
    generate_report(exp_data, pairwise_results, anova_results, report_path)
    
    # 打印关键结果
    print("\n" + "=" * 60)
    print("关键结果摘要")
    print("=" * 60)
    
    for name in exp_data:
        print(f"{name}: F1 = {exp_data[name]['mean_f1']:.4f} ± {exp_data[name]['std_f1']:.4f}")
    
    print(f"\nANOVA: F={anova_results['f_statistic']:.4f}, p={anova_results['p_value']:.4f}")
    print(f"{'显著差异存在！' if anova_results['significant'] else '无显著差异'}")
    
    print("\n" + "=" * 60)
    print(f"✓ 分析完成! 所有结果保存在: {output_dir.absolute()}")
    print(f"  - JSON结果: {json_path}")
    print(f"  - 可视化: {output_dir}/*.png")
    print(f"  - 完整报告: {report_path}")


if __name__ == '__main__':
    main()
