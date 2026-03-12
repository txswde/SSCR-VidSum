"""
完整的可解释性分析脚本
在训练完成后运行此脚本生成所有可解释性结果
"""

import torch
import numpy as np
import argparse
import os
from tqdm import tqdm

from models import Model_VideoSumm
from datasets import VideoSummDataset, my_collate_fn
from config import get_arguments
from torch.nn.utils.rnn import pad_sequence

# 导入可解释性工具
from analysis_causal import CausalAnalyzer
from interpretability.causal_explainer import CausalEffectAnalyzer, CausalVisualizer
from metrics.causal_metrics import (
    counterfactual_validity,
    feature_disentanglement_score,
    compute_ate_metrics
)


def load_model(checkpoint_path, args):
    """加载训练好的模型"""
    model = Model_VideoSumm(args=args)
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    print(f"✓ Loaded model from {checkpoint_path}")
    return model


def analyze_causal_effects(model, val_loader, args, num_samples=5):
    """分析因果效应 (原有工具)"""
    print("\n" + "="*60)
    print("1. 因果效应分析 (Causal Impact Analysis)")
    print("="*60)
    
    # 临时修改save_dir
    original_model_dir = args.model_dir
    args.model_dir = './interpretability_results'
    
    analyzer = CausalAnalyzer(
        model=model,
        args=args
    )
    
    # 恢复原始model_dir
    args.model_dir = original_model_dir
    
    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break
        
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
        
        # 计算因果影响矩阵
        impact_matrix, orig_probs = analyzer.compute_causal_effect(
            video=video,
            text=text,
            mask_video=mask_video,
            mask_text=mask_text,
            video_label=video_label,
            text_label=text_label,
            video_to_text_mask_list=video_to_text_mask_list,
            text_to_video_mask_list=text_to_video_mask_list
        )
        
        # 生成可视化
        for b in range(video.shape[0]):
            analyzer.visualize_and_save(
                impact_matrix[b],
                sample_idx=i*video.shape[0] + b,
                seq_idx=i*video.shape[0] + b
            )
    
    print(f"✓ Generated {num_samples} causal impact heatmaps")
    print(f"  Saved to: ./interpretability_results/causal_maps/")


def analyze_ate_cate(model, val_loader, args, num_samples=10):
    """分析ATE/CATE (新工具)"""
    print("\n" + "="*60)
    print("2. ATE/CATE 分析 (Average Treatment Effect)")
    print("="*60)
    
    ate_analyzer = CausalEffectAnalyzer(model=model)
    
    ate_min_list = []
    ate_sev_list = []
    ate_zero_list = []  # 零化干预
    effect_size_min_list = []
    effect_size_sev_list = []
    effect_size_zero_list = []
    
    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break
        
        video_list, text_list, mask_video_list, mask_text_list, \
        video_cls_label_list, _, _, text_cls_label_list, _, _, \
        _, _, _, _, _, _, video_to_text_mask_list, text_to_video_mask_list, \
        text_cf_min_list, text_cf_sev_list = batch
        
        video = pad_sequence(video_list, batch_first=True).to(args.device)
        text = pad_sequence(text_list, batch_first=True).to(args.device)
        text_cf_min = pad_sequence(text_cf_min_list, batch_first=True).to(args.device)
        text_cf_sev = pad_sequence(text_cf_sev_list, batch_first=True).to(args.device)
        text_zero = torch.zeros_like(text)  # 零化干预 - 完全移除文本信息
        
        mask_video = pad_sequence(mask_video_list, batch_first=True).to(args.device)
        mask_text = pad_sequence(mask_text_list, batch_first=True).to(args.device)
        video_label = pad_sequence(video_cls_label_list, batch_first=True).to(args.device)
        text_label = pad_sequence(text_cls_label_list, batch_first=True).to(args.device)
        
        for j in range(len(video_to_text_mask_list)):
            video_to_text_mask_list[j] = video_to_text_mask_list[j].to(args.device)
            text_to_video_mask_list[j] = text_to_video_mask_list[j].to(args.device)
        
        # 计算ATE (最小干预)
        ate_min = ate_analyzer.compute_ATE(
            video=video,
            text_orig=text,
            text_cf=text_cf_min,
            mask_video=mask_video,
            mask_text=mask_text,
            video_label=video_label,
            text_label=text_label,
            video_to_text_mask_list=video_to_text_mask_list,
            text_to_video_mask_list=text_to_video_mask_list
        )
        ate_min_list.append(ate_min['ATE'])
        effect_size_min_list.append(ate_min['effect_size'])
        
        # 计算ATE (强干预)
        ate_sev = ate_analyzer.compute_ATE(
            video=video,
            text_orig=text,
            text_cf=text_cf_sev,
            mask_video=mask_video,
            mask_text=mask_text,
            video_label=video_label,
            text_label=text_label,
            video_to_text_mask_list=video_to_text_mask_list,
            text_to_video_mask_list=text_to_video_mask_list
        )
        ate_sev_list.append(ate_sev['ATE'])
        effect_size_sev_list.append(ate_sev['effect_size'])
        
        # 计算ATE (零化干预 - 完全移除文本)
        ate_zero = ate_analyzer.compute_ATE(
            video=video,
            text_orig=text,
            text_cf=text_zero,
            mask_video=mask_video,
            mask_text=mask_text,
            video_label=video_label,
            text_label=text_label,
            video_to_text_mask_list=video_to_text_mask_list,
            text_to_video_mask_list=text_to_video_mask_list
        )
        ate_zero_list.append(ate_zero['ATE'])
        effect_size_zero_list.append(ate_zero['effect_size'])
    
    print(f"\n✓ ATE Analysis Results (n={num_samples}):")
    print(f"\n  [Minimal Intervention (噪声干预)]:")
    print(f"  Average ATE: {np.mean(ate_min_list):.4f} ± {np.std(ate_min_list):.4f}")
    print(f"  Average Effect Size: {np.mean(effect_size_min_list):.4f}")
    
    print(f"\n  [Severe Intervention (强噪声干预)]:")
    print(f"  Average ATE: {np.mean(ate_sev_list):.4f} ± {np.std(ate_sev_list):.4f}")
    print(f"  Average Effect Size: {np.mean(effect_size_sev_list):.4f}")
    
    print(f"\n  [Zero Intervention (完全零化文本)]:")
    print(f"  Average ATE: {np.mean(ate_zero_list):.4f} ± {np.std(ate_zero_list):.4f}")
    print(f"  Average Effect Size: {np.mean(effect_size_zero_list):.4f}")
    
    # 对比分析
    zero_effect = np.mean(np.abs(ate_zero_list))
    
    print(f"\n  [分析结论]:")
    if zero_effect > 0.05:
        print(f"  ✓ 模型依赖文本输入 (零化ATE={zero_effect:.4f})")
        print(f"    移除文本后预测显著改变,证明模型利用了文本-视频对齐信息")
    else:
        print(f"  ⚠️ 模型对文本输入不敏感 (零化ATE={zero_effect:.4f})")
        print(f"    可能原因: 1)视频特征主导决策 2)模型未学习好文本对齐")
    
    return ate_min_list, ate_sev_list, ate_zero_list


def analyze_feature_space(model, val_loader, args, num_samples=50):
    """特征空间可视化 (新工具)"""
    print("\n" + "="*60)
    print("3. 特征空间可视化 (t-SNE)")
    print("="*60)
    
    visualizer = CausalVisualizer(save_dir='./interpretability_results')
    
    feat_orig_list = []
    feat_cf_min_list = []
    feat_cf_sev_list = []
    causal_feat_list = []
    spurious_feat_list = []
    
    for i, batch in enumerate(tqdm(val_loader, desc="Collecting features")):
        if i >= num_samples:
            break
        
        video_list, text_list, mask_video_list, mask_text_list, \
        video_cls_label_list, _, _, text_cls_label_list, _, _, \
        _, _, _, _, _, _, video_to_text_mask_list, text_to_video_mask_list, \
        text_cf_min_list, text_cf_sev_list = batch
        
        video = pad_sequence(video_list, batch_first=True).to(args.device)
        text = pad_sequence(text_list, batch_first=True).to(args.device)
        text_cf_min = pad_sequence(text_cf_min_list, batch_first=True).to(args.device)
        text_cf_sev = pad_sequence(text_cf_sev_list, batch_first=True).to(args.device)
        mask_video = pad_sequence(mask_video_list, batch_first=True).to(args.device)
        mask_text = pad_sequence(mask_text_list, batch_first=True).to(args.device)
        video_label = pad_sequence(video_cls_label_list, batch_first=True).to(args.device)
        text_label = pad_sequence(text_cls_label_list, batch_first=True).to(args.device)
        
        for j in range(len(video_to_text_mask_list)):
            video_to_text_mask_list[j] = video_to_text_mask_list[j].to(args.device)
            text_to_video_mask_list[j] = text_to_video_mask_list[j].to(args.device)
        
        with torch.no_grad():
            outputs = model(
                video=video,
                text=text,
                mask_video=mask_video,
                mask_text=mask_text,
                video_label=video_label,
                text_label=text_label,
                video_to_text_mask_list=video_to_text_mask_list,
                text_to_video_mask_list=text_to_video_mask_list,
                text_cf_minimal=text_cf_min,
                text_cf_severe=text_cf_sev
            )
        
        # 收集特征 (取平均池化)
        feat_orig_list.append(outputs['text_feat_orig'].mean(dim=1).cpu().numpy())
        if outputs['text_feat_cf_min'] is not None:
            feat_cf_min_list.append(outputs['text_feat_cf_min'].mean(dim=1).cpu().numpy())
        if outputs['text_feat_cf_sev'] is not None:
            feat_cf_sev_list.append(outputs['text_feat_cf_sev'].mean(dim=1).cpu().numpy())
        causal_feat_list.append(outputs['causal_feat'].mean(dim=1).cpu().numpy())
        spurious_feat_list.append(outputs['spurious_feat'].mean(dim=1).cpu().numpy())
    
    # 合并特征
    feat_orig = np.concatenate(feat_orig_list, axis=0)
    feat_cf_min = np.concatenate(feat_cf_min_list, axis=0) if feat_cf_min_list else None
    feat_cf_sev = np.concatenate(feat_cf_sev_list, axis=0) if feat_cf_sev_list else None
    
    # 生成t-SNE可视化
    if feat_cf_min is not None and feat_cf_sev is not None:
        visualizer.plot_feature_space(
            feat_orig=feat_orig,
            feat_cf_min=feat_cf_min,
            feat_cf_sev=feat_cf_sev,
            save_name='feature_space_tsne.png'
        )
        print(f"✓ Generated t-SNE visualization")
        print(f"  Saved to: ./interpretability_results/feature_space_tsne.png")
    
    # 计算特征解耦分数
    causal_feat = np.concatenate(causal_feat_list, axis=0)
    spurious_feat = np.concatenate(spurious_feat_list, axis=0)
    
    # 转换回tensor计算FDS
    causal_feat_tensor = torch.from_numpy(causal_feat).unsqueeze(1)
    spurious_feat_tensor = torch.from_numpy(spurious_feat).unsqueeze(1)
    
    fds_results = feature_disentanglement_score(causal_feat_tensor, spurious_feat_tensor)
    
    print(f"\n✓ Feature Disentanglement Score:")
    print(f"  FDS: {fds_results['FDS']:.4f}")
    print(f"  Avg Correlation: {fds_results['avg_correlation']:.4f}")
    print(f"  Is Disentangled: {bool(fds_results['is_disentangled'])}")


def main():
    # 解析参数
    args = get_arguments()
    
    # 设置保存目录
    os.makedirs('./interpretability_results', exist_ok=True)
    os.makedirs('./interpretability_results/causal_maps', exist_ok=True)
    
    print("="*60)
    print("可解释性分析工具")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    
    # 加载数据
    import h5py
    import yaml
    
    split_path = f'{args.data_root}/{args.dataset}/splits.yml'
    with open(split_path, 'r') as f:
        splits = yaml.safe_load(f)
    
    # 使用第一个split的测试集
    val_keys = splits[0]['test_keys']
    val_dataset = VideoSummDataset(val_keys, args=args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=my_collate_fn
    )
    
    # 加载最佳模型
    checkpoint_path = f'{args.model_dir}/checkpoint/model_best_split0.pt'
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"\nSearching in: {args.model_dir}/checkpoint/")
        if os.path.exists(f'{args.model_dir}/checkpoint/'):
            files = os.listdir(f'{args.model_dir}/checkpoint/')
            print(f"Available files: {files}")
        print("\nPlease train the model first!")
        return
    
    model = load_model(checkpoint_path, args)
    
    # 运行所有分析
    analyze_causal_effects(model, val_loader, args, num_samples=3)
    analyze_ate_cate(model, val_loader, args, num_samples=10)
    analyze_feature_space(model, val_loader, args, num_samples=50)
    
    print("\n" + "="*60)
    print("✅ 所有可解释性分析完成!")
    print("="*60)
    print("\n查看结果:")
    print("  1. 因果热力图: ./interpretability_results/causal_maps/")
    print("  2. t-SNE可视化: ./interpretability_results/feature_space_tsne.png")
    print("  3. ATE统计: 见上方输出")


if __name__ == '__main__':
    main()
