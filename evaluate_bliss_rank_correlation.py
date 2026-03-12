
"""
BLiSS模型秩相关系数评测脚本

评估logs/BLiSS目录下四个模型的预测结果与真实标签的秩相关系数：
1. BLiSS_supervised_baseline
2. BLiSS_unsupervised  
3. BLiSS_unsupervised_yinguo
4. BLiSS_unsupervised_yinguo_duiqi_corrected

使用Spearman秩相关系数评估模型预测与ground truth的一致性
"""

import os
import json
import argparse
import torch
import numpy as np
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm
from collections import defaultdict

# 导入项目模块
from config import build_args
from models import Model_BLiSS
from datasets import BLiSSDataset, my_collate_fn, worker_init_fn


def create_args_for_bliss(data_root='./data'):
    """为BLiSS数据集创建配置参数"""
    import argparse
    
    # BLiSS数据集的默认参数
    args = argparse.Namespace(
        dataset='BLiSS',
        data_root=data_root,
        num_input_video=512,
        num_input_text=768,
        num_hidden=128,
        num_layers=6,
        dropout_video=0.2,
        dropout_text=0.2,
        dropout_attn=0.2,
        dropout_fc=0.2,
        ratio=4,
    )
    return args


def load_model(args, checkpoint_path: str):
    """加载训练好的模型"""
    model = Model_BLiSS(args=args)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理不同格式的checkpoint
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 尝试加载，使用strict=False忽略不匹配的键
        try:
            model.load_state_dict(state_dict, strict=False)
            print(f"✓ 加载模型: {checkpoint_path}")
        except Exception as e:
            print(f"⚠ 加载模型警告: {e}")
            # 尝试移除前缀 "module." (如果是DataParallel保存的)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
            print(f"✓ 加载模型(移除module前缀后): {checkpoint_path}")
    else:
        print(f"✗ 模型文件不存在: {checkpoint_path}")
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return model


def compute_rank_correlation(pred_scores, gt_labels, mask=None):
    """
    计算预测分数与真实标签的秩相关系数
    
    Args:
        pred_scores: 预测分数 (可能是tensor或numpy数组)
        gt_labels: 真实标签 (0/1)
        mask: 有效位置掩码
        
    Returns:
        dict: 包含Spearman和Kendall相关系数
    """
    # 转换为numpy
    if torch.is_tensor(pred_scores):
        pred_scores = pred_scores.detach().cpu().numpy()
    if torch.is_tensor(gt_labels):
        gt_labels = gt_labels.detach().cpu().numpy()
    if mask is not None and torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    
    # 应用mask
    if mask is not None:
        valid_idx = mask.astype(bool)
        pred_scores = pred_scores[valid_idx]
        gt_labels = gt_labels[valid_idx]
    
    # 过滤无效值
    if len(pred_scores) < 2 or len(gt_labels) < 2:
        return {'spearman_rho': np.nan, 'spearman_p': np.nan,
                'kendall_tau': np.nan, 'kendall_p': np.nan}
    
    # 检查是否有有效变化
    if np.std(pred_scores) == 0 or np.std(gt_labels) == 0:
        return {'spearman_rho': np.nan, 'spearman_p': np.nan,
                'kendall_tau': np.nan, 'kendall_p': np.nan}
    
    try:
        spearman_rho, spearman_p = spearmanr(pred_scores, gt_labels)
        kendall_tau, kendall_p = kendalltau(pred_scores, gt_labels)
    except Exception as e:
        print(f"计算相关系数时出错: {e}")
        return {'spearman_rho': np.nan, 'spearman_p': np.nan,
                'kendall_tau': np.nan, 'kendall_p': np.nan}
    
    return {
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_p': kendall_p
    }


def evaluate_model(model, dataloader, device):
    """
    评估单个模型的秩相关系数
    
    Returns:
        dict: 包含video和text的秩相关结果
    """
    all_video_corr = []
    all_text_corr = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            # 解包数据
            video_list, video_summ_list, text_list, mask_video_list, mask_video_summ_list, \
                mask_text_list, video_label_list, text_label_list, sentence_list, \
                highlight_list, video_to_text_mask_list, text_to_video_mask_list = batch
            
            # 批量处理
            for i in range(len(video_list)):
                video = video_list[i].unsqueeze(0).to(device)
                text = text_list[i].unsqueeze(0).to(device)
                mask_video = mask_video_list[i].unsqueeze(0).to(device)
                mask_text = mask_text_list[i].unsqueeze(0).to(device)
                video_label = video_label_list[i].unsqueeze(0).to(device)
                text_label = text_label_list[i].unsqueeze(0).to(device)
                video_to_text_mask = [video_to_text_mask_list[i].to(device)]
                text_to_video_mask = [text_to_video_mask_list[i].to(device)]
                
                # 模型推理
                try:
                    pred_video, pred_text, _ = model(
                        video=video,
                        text=text,
                        mask_video=mask_video,
                        mask_text=mask_text,
                        video_label=video_label,
                        text_label=text_label,
                        video_to_text_mask_list=video_to_text_mask,
                        text_to_video_mask_list=text_to_video_mask
                    )
                except Exception as e:
                    print(f"模型推理错误: {e}")
                    continue
                
                # 计算video秩相关
                video_corr = compute_rank_correlation(
                    pred_video[0], video_label[0], mask_video[0]
                )
                if not np.isnan(video_corr['spearman_rho']):
                    all_video_corr.append(video_corr)
                
                # 计算text秩相关
                text_corr = compute_rank_correlation(
                    pred_text[0], text_label[0], mask_text[0]
                )
                if not np.isnan(text_corr['spearman_rho']):
                    all_text_corr.append(text_corr)
    
    # 汇总结果
    def aggregate_results(corr_list):
        if len(corr_list) == 0:
            return {
                'spearman_rho_mean': np.nan, 'spearman_rho_std': np.nan,
                'kendall_tau_mean': np.nan, 'kendall_tau_std': np.nan,
                'n_samples': 0
            }
        
        spearman_rhos = [c['spearman_rho'] for c in corr_list]
        kendall_taus = [c['kendall_tau'] for c in corr_list]
        
        return {
            'spearman_rho_mean': np.mean(spearman_rhos),
            'spearman_rho_std': np.std(spearman_rhos),
            'kendall_tau_mean': np.mean(kendall_taus),
            'kendall_tau_std': np.std(kendall_taus),
            'n_samples': len(corr_list)
        }
    
    return {
        'video': aggregate_results(all_video_corr),
        'text': aggregate_results(all_text_corr)
    }


def main():
    parser = argparse.ArgumentParser(description='BLiSS模型秩相关系数评测')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--logs_dir', type=str, default='./logs/BLiSS')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # 获取默认配置
    config_args = create_args_for_bliss(args.data_root)
    
    # 四个模型目录
    model_dirs = [
        'BLiSS_supervised_baseline',
        'BLiSS_unsupervised',
        'BLiSS_unsupervised_yinguo',
        'BLiSS_unsupervised_yinguo_duiqi_corrected'
    ]
    
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 加载测试数据集
    print("\n加载测试数据集...")
    test_dataset = BLiSSDataset(mode='test', args=config_args)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=my_collate_fn,
        worker_init_fn=worker_init_fn
    )
    print(f"测试样本数: {len(test_dataset)}")
    
    # 评估每个模型
    results = {}
    for model_name in model_dirs:
        print(f"\n{'='*60}")
        print(f"评估模型: {model_name}")
        print('='*60)
        
        # 查找模型检查点
        model_dir = os.path.join(args.logs_dir, model_name)
        checkpoint_paths = [
            os.path.join(model_dir, 'model_best_video.pt'),
            os.path.join(model_dir, 'model_best_text.pt'),
            os.path.join(model_dir, 'checkpoint', 'model_best.pt'),
            # 添加子目录中的检查点路径
            os.path.join(model_dir, 'checkpoint', 'model_best_video.pt'),
            os.path.join(model_dir, 'checkpoint', 'model_best_text.pt'),
        ]
        
        checkpoint_path = None
        for cp in checkpoint_paths:
            if os.path.exists(cp):
                checkpoint_path = cp
                break
        
        if checkpoint_path is None:
            print(f"未找到模型检查点，跳过: {model_name}")
            results[model_name] = None
            continue
        
        # 加载模型
        model = load_model(config_args, checkpoint_path)
        if model is None:
            results[model_name] = None
            continue
        
        model = model.to(device)
        
        # 评估
        model_results = evaluate_model(model, test_loader, device)
        results[model_name] = model_results
        
        # 打印结果
        print(f"\n{model_name} 结果:")
        print("-" * 40)
        print("Video 预测:")
        print(f"  Spearman ρ: {model_results['video']['spearman_rho_mean']:.4f} ± {model_results['video']['spearman_rho_std']:.4f}")
        print(f"  Kendall τ:  {model_results['video']['kendall_tau_mean']:.4f} ± {model_results['video']['kendall_tau_std']:.4f}")
        print(f"  样本数:     {model_results['video']['n_samples']}")
        print()
        print("Text 预测:")
        print(f"  Spearman ρ: {model_results['text']['spearman_rho_mean']:.4f} ± {model_results['text']['spearman_rho_std']:.4f}")
        print(f"  Kendall τ:  {model_results['text']['kendall_tau_mean']:.4f} ± {model_results['text']['kendall_tau_std']:.4f}")
        print(f"  样本数:     {model_results['text']['n_samples']}")
        
        # 释放GPU内存
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("秩相关系数汇总表")
    print("=" * 80)
    
    # 打印表头
    print(f"{'模型名称':<45} | {'Video Spearman ρ':^18} | {'Text Spearman ρ':^18}")
    print("-" * 80)
    
    for model_name in model_dirs:
        if results[model_name] is not None:
            v_rho = results[model_name]['video']['spearman_rho_mean']
            v_std = results[model_name]['video']['spearman_rho_std']
            t_rho = results[model_name]['text']['spearman_rho_mean']
            t_std = results[model_name]['text']['spearman_rho_std']
            print(f"{model_name:<45} | {v_rho:>7.4f} ± {v_std:<7.4f} | {t_rho:>7.4f} ± {t_std:<7.4f}")
        else:
            print(f"{model_name:<45} | {'N/A':^18} | {'N/A':^18}")
    
    print("=" * 80)
    
    # 保存结果到JSON
    output_path = os.path.join(args.logs_dir, 'rank_correlation_results.json')
    
    # 转换NaN为None以便JSON序列化
    def convert_nan(obj):
        if isinstance(obj, dict):
            return {k: convert_nan(v) for k, v in obj.items()}
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return obj
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_nan(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
