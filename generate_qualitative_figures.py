"""
定性案例三行图生成脚本 (Qualitative Case 3-Row Figure Generator)

生成两种数据集的定性分析图：
1. BLiSS 数据集 - 1个案例
2. TVSum 数据集 - 1个案例

每张图包含三行:
- Row 1: 重要性曲线 (Importance Curve) - 帧级预测分数
- Row 2: 对齐热力图 (Alignment Heatmap) - 句子-帧对齐矩阵
- Row 3: 干预效应条形图 (Intervention Effect Bar) - 每句话的因果效应

Author: Auto-generated for Paper Assets
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import build_args
from models import Model_BLiSS, Model_VideoSumm
from datasets import BLiSSDataset, VideoSummDataset
import h5py

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置论文级别的样式
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def load_bliss_model(model_dir, args):
    """加载 BLiSS 模型"""
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
    
    return model


def load_tvsum_model(model_dir, args):
    """加载 TVSum 模型"""
    model = Model_VideoSumm(args=args)
    
    checkpoint_paths = [
        os.path.join(model_dir, 'checkpoint', 'model_best.pt'),
        os.path.join(model_dir, 'model_best.pt'),
        os.path.join(model_dir, 'checkpoint', 'model_best_split0.pt'),  # 使用第一个 split
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


def compute_bliss_case_data(model, sample, args):
    """计算 BLiSS 单个案例的所有数据"""
    video, video_summ, text, mask_video, mask_video_summ, mask_text, \
        video_label, text_label, sentence, highlight, \
        video_to_text_mask, text_to_video_mask = sample
    
    num_frames = video.shape[0]
    num_sentences = text.shape[0]
    
    # Create batch
    video_b = video.unsqueeze(0).to(args.device)
    text_b = text.unsqueeze(0).to(args.device)
    mask_video_b = mask_video.unsqueeze(0).to(args.device)
    mask_text_b = mask_text.unsqueeze(0).to(args.device)
    video_label_b = video_label.unsqueeze(0).to(args.device)
    text_label_b = text_label.unsqueeze(0).to(args.device)
    video_to_text_mask_list = [video_to_text_mask.to(args.device)]
    text_to_video_mask_list = [text_to_video_mask.to(args.device)]
    
    with torch.no_grad():
        # Original prediction
        pred_video_orig, pred_text_orig, _ = model(
            video=video_b, text=text_b,
            mask_video=mask_video_b, mask_text=mask_text_b,
            video_label=video_label_b, text_label=text_label_b,
            video_to_text_mask_list=video_to_text_mask_list,
            text_to_video_mask_list=text_to_video_mask_list
        )
        
        # Row 1: Importance scores
        importance_scores = torch.sigmoid(pred_video_orig).squeeze().cpu().numpy()[:num_frames]
        gt_video_label = video_label.numpy()[:num_frames]
        
        # Row 2 & 3: Compute causal effect matrix via intervention
        causal_matrix = torch.zeros(num_frames, num_sentences)
        intervention_effects = []
        
        for sent_idx in range(num_sentences):
            # Mask out sentence sent_idx
            text_masked = text_b.clone()
            text_masked[0, sent_idx, :] = 0
            
            pred_video_masked, _, _ = model(
                video=video_b, text=text_masked,
                mask_video=mask_video_b, mask_text=mask_text_b,
                video_label=video_label_b, text_label=text_label_b,
                video_to_text_mask_list=video_to_text_mask_list,
                text_to_video_mask_list=text_to_video_mask_list
            )
            
            effect = (torch.sigmoid(pred_video_orig) - torch.sigmoid(pred_video_masked)).abs().squeeze()
            causal_matrix[:, sent_idx] = effect[:num_frames].cpu()
            intervention_effects.append(effect[:num_frames].sum().item())
    
    # Normalize causal matrix
    causal_np = causal_matrix.numpy()
    if causal_np.max() > 0:
        causal_np = causal_np / causal_np.max()
    
    # Get GT alignment for comparison
    gt_alignment = video_to_text_mask.cpu().numpy()
    
    return {
        'importance_scores': importance_scores,
        'gt_video_label': gt_video_label,
        'causal_matrix': causal_np,
        'gt_alignment': gt_alignment,
        'intervention_effects': np.array(intervention_effects),
        'sentences': sentence,
        'num_frames': num_frames,
        'num_sentences': num_sentences,
    }


def compute_tvsum_case_data(model, video, text, args, video_name):
    """计算 TVSum 单个案例的所有数据"""
    import math
    
    num_frames = video.shape[0]
    num_sentences = text.shape[0]
    
    video_b = video.unsqueeze(0).to(args.device)
    text_b = text.unsqueeze(0).to(args.device)
    mask_video = torch.ones(1, num_frames, dtype=torch.long).to(args.device)
    mask_text = torch.ones(1, num_sentences, dtype=torch.long).to(args.device)
    
    # Create dummy labels (needed by forward but not used in inference)
    video_label = torch.zeros(1, num_frames, dtype=torch.long).to(args.device)
    text_label = torch.zeros(1, num_sentences, dtype=torch.long).to(args.device)
    
    # Create time-based alignment masks (diagonal-like)
    frame_sentence_ratio = int(math.ceil(num_frames / num_sentences))
    video_to_text_mask = torch.zeros((num_frames, num_sentences), dtype=torch.long)
    text_to_video_mask = torch.zeros((num_sentences, num_frames), dtype=torch.long)
    for j in range(num_sentences):
        start_frame = j * frame_sentence_ratio
        end_frame = min((j + 1) * frame_sentence_ratio, num_frames)
        video_to_text_mask[start_frame:end_frame, j] = 1
        text_to_video_mask[j, start_frame:end_frame] = 1
    
    video_to_text_mask_list = [video_to_text_mask.to(args.device)]
    text_to_video_mask_list = [text_to_video_mask.to(args.device)]
    
    with torch.no_grad():
        # Original prediction
        outputs = model(
            video=video_b, text=text_b, 
            mask_video=mask_video, mask_text=mask_text,
            video_label=video_label, text_label=text_label,
            video_to_text_mask_list=video_to_text_mask_list,
            text_to_video_mask_list=text_to_video_mask_list
        )
        
        if isinstance(outputs, dict):
             # For Model_VideoSumm which returns a dict
             pred_video_orig = outputs['video_pred_cls']
        else:
             # Fallback for tuple or tensor
             pred_video_orig = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Row 1: Importance scores
        importance_scores = torch.sigmoid(pred_video_orig).squeeze().cpu().numpy()[:num_frames]
        
        # Row 2 & 3: Compute causal effect matrix via intervention
        causal_matrix = torch.zeros(num_frames, num_sentences)
        intervention_effects = []
        
        for sent_idx in range(num_sentences):
            text_masked = text_b.clone()
            text_masked[0, sent_idx, :] = 0
            
            outputs_masked = model(
                video=video_b, text=text_masked, 
                mask_video=mask_video, mask_text=mask_text,
                video_label=video_label, text_label=text_label,
                video_to_text_mask_list=video_to_text_mask_list,
                text_to_video_mask_list=text_to_video_mask_list
            )
            
            if isinstance(outputs_masked, dict):
                 pred_video_masked = outputs_masked['video_pred_cls']
            else:
                 pred_video_masked = outputs_masked[0] if isinstance(outputs_masked, tuple) else outputs_masked
            
            effect = (torch.sigmoid(pred_video_orig) - torch.sigmoid(pred_video_masked)).abs().squeeze()
            causal_matrix[:, sent_idx] = effect[:num_frames].cpu()
            intervention_effects.append(effect[:num_frames].sum().item())
    
    # Normalize causal matrix
    causal_np = causal_matrix.numpy()
    if causal_np.max() > 0:
        causal_np = causal_np / causal_np.max()
    
    # Load sentence text
    caption_path = f'data/TVSum/caption/{video_name}.txt'
    sentences = []
    if os.path.exists(caption_path):
        with open(caption_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
    
    # Store GT alignment for visualization
    gt_alignment = video_to_text_mask.cpu().numpy()
    
    return {
        'importance_scores': importance_scores,
        'gt_video_label': None,  # TVSum GT is in user_summary, will handle separately
        'causal_matrix': causal_np,
        'gt_alignment': gt_alignment,  # Use time-based alignment
        'intervention_effects': np.array(intervention_effects),
        'sentences': sentences[:num_sentences],  # Match feature count
        'num_frames': num_frames,
        'num_sentences': num_sentences,
    }


def plot_three_row_figure(case_data, title, save_path, show_gt=True):
    """
    绘制三行定性分析图
    
    Row 1: Importance Curve (帧重要性曲线)
    Row 2: Alignment Heatmap (对齐热力图)  
    Row 3: Intervention Effect Bar (干预效应条形图)
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 1, height_ratios=[1, 1.5, 1], hspace=0.3)
    
    num_frames = case_data['num_frames']
    num_sentences = case_data['num_sentences']
    importance = case_data['importance_scores']
    causal_matrix = case_data['causal_matrix']
    intervention_effects = case_data['intervention_effects']
    sentences = case_data['sentences']
    
    # === Row 1: Importance Curve ===
    ax1 = fig.add_subplot(gs[0])
    frames = np.arange(num_frames)
    
    ax1.fill_between(frames, importance, alpha=0.3, color='steelblue', label='Predicted Importance')
    ax1.plot(frames, importance, color='steelblue', linewidth=1.5)
    
    # Add GT if available
    if show_gt and case_data['gt_video_label'] is not None:
        gt = case_data['gt_video_label'].astype(float)
        # Highlight GT keyframes
        ax1.fill_between(frames, gt * importance.max(), alpha=0.2, color='red', label='GT Keyframes')
    
    ax1.set_xlim(0, num_frames - 1)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('Frame Index', fontsize=11)
    ax1.set_ylabel('Importance Score ($S_t$)', fontsize=11)
    ax1.set_title('(a) Frame-level Importance Prediction', fontsize=12, fontweight='bold', loc='left')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # === Row 2: Alignment Heatmap ===
    ax2 = fig.add_subplot(gs[1])
    
    # Transpose for better visualization: x=frame, y=sentence
    heatmap_data = causal_matrix.T  # [num_sentences, num_frames]
    
    im = ax2.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', 
                    extent=[0, num_frames, num_sentences, 0])
    
    # Add GT alignment contour if available
    if show_gt and case_data['gt_alignment'] is not None:
        gt_align = case_data['gt_alignment'].T  # [num_sentences, num_frames]
        ax2.contour(np.arange(num_frames) + 0.5, np.arange(num_sentences) + 0.5, 
                   gt_align, levels=[0.5], colors='cyan', linewidths=1.5, linestyles='--')
    
    ax2.set_xlabel('Frame Index', fontsize=11)
    ax2.set_ylabel('Sentence Index', fontsize=11)
    ax2.set_title('(b) Sentence-Frame Alignment Matrix ($A_{j,t}$)', fontsize=12, fontweight='bold', loc='left')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.02, pad=0.02)
    cbar.set_label('Causal Effect', fontsize=10)
    
    # === Row 3: Intervention Effect Bar ===
    ax3 = fig.add_subplot(gs[2])
    
    sent_indices = np.arange(num_sentences)
    colors = plt.cm.viridis(intervention_effects / (intervention_effects.max() + 1e-8))
    
    bars = ax3.bar(sent_indices, intervention_effects, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add sentence labels (truncated)
    if sentences:
        truncated_labels = []
        for i, sent in enumerate(sentences[:num_sentences]):
            if isinstance(sent, str):
                label = sent[:25] + '...' if len(sent) > 25 else sent
            else:
                label = f'Sent {i}'
            truncated_labels.append(label)
        ax3.set_xticks(sent_indices)
        ax3.set_xticklabels(truncated_labels, rotation=45, ha='right', fontsize=8)
    else:
        ax3.set_xticks(sent_indices)
        ax3.set_xticklabels([f'S{i}' for i in sent_indices])
    
    ax3.set_xlabel('Sentence', fontsize=11)
    ax3.set_ylabel('Total Intervention Effect', fontsize=11)
    ax3.set_title('(c) Causal Effect of Masking Each Sentence', fontsize=12, fontweight='bold', loc='left')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Main title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ 图片已保存: {save_path}")
    return save_path


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("定性案例三行图生成器 (Qualitative Case 3-Row Figure Generator)")
    print("=" * 70)
    
    # 创建保存目录
    save_dir = './interpretability_results/paper_assets'
    os.makedirs(save_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # ========== BLiSS Case ==========
    print("\n" + "-" * 50)
    print("[1/2] 处理 BLiSS 数据集...")
    print("-" * 50)
    
    try:
        # Build BLiSS args
        old_argv = sys.argv
        sys.argv = ['', '--dataset=BLiSS']
        bliss_args = build_args()
        sys.argv = old_argv
        bliss_args.device = device
        
        # Load BLiSS dataset
        print("  加载数据集...")
        bliss_test = BLiSSDataset(mode='test', args=bliss_args)
        print(f"    ✓ 共 {len(bliss_test)} 个测试样本")
        
        # Load BLiSS model (use the best model)
        bliss_model_dir = './logs/BLiss/BLiSS_unsupervised_yinguo_duiqi_corrected'
        if not os.path.exists(bliss_model_dir):
            # Try alternative paths
            alt_dirs = [
                './logs/BLiss/BLiSS_unsupervised',
                './logs/BLiss/BLiSS_supervised_baseline',
            ]
            for alt in alt_dirs:
                if os.path.exists(alt):
                    bliss_model_dir = alt
                    break
        
        print(f"  加载模型: {bliss_model_dir}")
        bliss_args.model_dir = bliss_model_dir
        bliss_model = load_bliss_model(bliss_model_dir, bliss_args)
        print("    ✓ 模型加载成功")
        
        # Select a good sample (one with clear alignment)
        sample_idx = 0  # Use first sample, can be changed
        sample = bliss_test[sample_idx]
        
        print(f"  分析样本 {sample_idx}...")
        bliss_case = compute_bliss_case_data(bliss_model, sample, bliss_args)
        
        # Generate figure
        bliss_save_path = os.path.join(save_dir, 'qualitative_case_bliss.png')
        plot_three_row_figure(
            bliss_case, 
            title='BLiSS Dataset - Qualitative Case Analysis',
            save_path=bliss_save_path,
            show_gt=True
        )
        
        # Save raw data for future use
        bliss_data_path = os.path.join(save_dir, 'bliss_case_data.json')
        with open(bliss_data_path, 'w', encoding='utf-8') as f:
            json.dump({
                'importance_scores': bliss_case['importance_scores'].tolist(),
                'intervention_effects': bliss_case['intervention_effects'].tolist(),
                'sentences': bliss_case['sentences'] if isinstance(bliss_case['sentences'], list) else list(bliss_case['sentences']),
                'num_frames': bliss_case['num_frames'],
                'num_sentences': bliss_case['num_sentences'],
            }, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 原始数据已保存: {bliss_data_path}")
        
    except Exception as e:
        print(f"  ⚠️ BLiSS 处理失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== TVSum Case ==========
    print("\n" + "-" * 50)
    print("[2/2] 处理 TVSum 数据集...")
    print("-" * 50)
    
    try:
        # Build TVSum args
        old_argv = sys.argv
        sys.argv = ['', '--dataset=TVSum']
        tvsum_args = build_args()
        sys.argv = old_argv
        tvsum_args.device = device
        
        # Load TVSum model
        tvsum_model_dir = './logs/TVSum_full_causal'
        
        print(f"  加载模型: {tvsum_model_dir}")
        tvsum_args.model_dir = tvsum_model_dir
        tvsum_model = load_tvsum_model(tvsum_model_dir, tvsum_args)
        print("    ✓ 模型加载成功")
        
        # Load TVSum data directly
        h5_path = f'{tvsum_args.data_root}/TVSum/feature/eccv16_dataset_tvsum_google_pool5.h5'
        text_path = f'{tvsum_args.data_root}/TVSum/feature/text_roberta.npy'
        
        print("  加载数据...")
        video_h5 = h5py.File(h5_path, 'r')
        text_dict = np.load(text_path, allow_pickle=True).item()
        
        # Select a video
        video_names = list(video_h5.keys())
        video_name = video_names[0]  # Use first video
        
        video_file = video_h5[video_name]
        video = torch.from_numpy(video_file['features'][...].astype(np.float32))
        text = torch.from_numpy(text_dict[video_name]).to(torch.float32)
        
        print(f"  分析视频: {video_name}")
        tvsum_case = compute_tvsum_case_data(tvsum_model, video, text, tvsum_args, video_name)
        
        # Add gtscore if available
        if 'gtscore' in video_file:
            gtscore = video_file['gtscore'][...].astype(np.float32)
            gtscore = (gtscore - gtscore.min()) / (gtscore.max() - gtscore.min() + 1e-8)
            # Binarize for visualization
            tvsum_case['gt_video_label'] = (gtscore > gtscore.mean()).astype(float)
        
        video_h5.close()
        
        # Generate figure
        tvsum_save_path = os.path.join(save_dir, 'qualitative_case_tvsum.png')
        plot_three_row_figure(
            tvsum_case,
            title=f'TVSum Dataset - Qualitative Case Analysis ({video_name})',
            save_path=tvsum_save_path,
            show_gt=True
        )
        
        # Save raw data
        tvsum_data_path = os.path.join(save_dir, 'tvsum_case_data.json')
        with open(tvsum_data_path, 'w', encoding='utf-8') as f:
            json.dump({
                'importance_scores': tvsum_case['importance_scores'].tolist(),
                'intervention_effects': tvsum_case['intervention_effects'].tolist(),
                'sentences': tvsum_case['sentences'],
                'num_frames': tvsum_case['num_frames'],
                'num_sentences': tvsum_case['num_sentences'],
                'video_name': video_name,
            }, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 原始数据已保存: {tvsum_data_path}")
        
    except Exception as e:
        print(f"  ⚠️ TVSum 处理失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"所有图片已保存至: {save_dir}")
    print("=" * 70)
    
    # List generated files
    print("\n生成的文件:")
    for f in os.listdir(save_dir):
        print(f"  • {f}")


if __name__ == '__main__':
    main()
