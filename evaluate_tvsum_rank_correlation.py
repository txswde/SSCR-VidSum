"""
TVSum Rank Correlation Evaluation Script (Fixed)
================================================
Evaluates the Spearman and Kendall rank correlation coefficients for a trained TVSum model.
It compares the model's predicted frame importance scores against the ground truth **continuous** gtscore.

IMPORTANT FIX: 
- Previous version incorrectly used binary user_summary (0/1) as GT
- This version uses continuous gtscore from original annotations

Metric:
- Average Spearman's rho
- Average Kendall's tau
Calculated by comparing the predicted score profile directly at the subsampled 
level against the continuous ground truth importance scores.
"""

import os
import sys
import argparse
import numpy as np
import h5py
import torch
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import build_args
from models import Model_VideoSumm
from helpers.vsumm_helper import get_keyshot_summ


def upsample_scores_to_full_frames(scores, picks, n_frames):
    """
    Upsample downsampled scores to full frame rate.
    
    Args:
        scores: Importance scores at subsampled positions [T]
        picks: Positions of subsampled frames in original video [T]
        n_frames: Total number of frames in original video
        
    Returns:
        Upsampled scores at full frame rate [n_frames]
    """
    picks = np.asarray(picks, dtype=np.int32)
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    
    for i in range(len(picks)):
        pos_lo = picks[i]
        pos_hi = picks[i + 1] if i + 1 < len(picks) else n_frames
        frame_scores[pos_lo:pos_hi] = scores[i]
    
    return frame_scores


def evaluate_rank_correlation(args):
    print(f"\n{'='*60}")
    print(f"Evaluating Rank Correlation for {args.dataset}")
    print(f"Model Directory: {args.model_dir}")
    print(f"Using CONTINUOUS gtscore (fixed evaluation)")
    print(f"{'='*60}")

    device = torch.device(args.device)
    
    # Load h5 file directly for gtscore access
    h5_path = f'data/{args.dataset}/feature/eccv16_dataset_{args.dataset.lower()}_google_pool5.h5'
    video_h5 = h5py.File(h5_path, 'r')
    
    # Load text features
    text_feature_path = f'data/{args.dataset}/feature/text_roberta.npy'
    text_feature_dict = np.load(text_feature_path, allow_pickle=True).item()
    
    import yaml
    split_path = f'data/{args.dataset}/splits.yml'
    with open(split_path, 'r') as f:
        splits = yaml.safe_load(f)
        
    num_splits = len(splits)
    print(f"Found {num_splits} splits in {split_path}")
    
    split_spearman = []
    split_kendall = []
    
    # Iterate over all splits (5-fold)
    for split_idx in range(num_splits):
        print(f"\nProcessing Split {split_idx}...")
        
        # 1. Load Model for current split
        model = Model_VideoSumm(args=args)
        
        checkpoint_paths = [
            os.path.join(args.model_dir, 'checkpoint', f'model_best_split{split_idx}.pt'),
            os.path.join(args.model_dir, f'model_best_split{split_idx}.pt'),
            args.checkpoint if args.checkpoint and split_idx == 0 else None
        ]
        
        checkpoint_path = None
        for cp in checkpoint_paths:
            if cp and os.path.exists(cp):
                checkpoint_path = cp
                break
                
        if not checkpoint_path:
            print(f"⚠️  No checkpoint found for Split {split_idx}. Skipping.")
            continue

        print(f"  Loading: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            continue

        # 2. Extract Test Keys for current split
        test_keys = splits[split_idx]['test_keys']
        print(f"  Test samples: {len(test_keys)}")
        
        all_spearman = []
        all_kendall = []

        with torch.no_grad():
            for key in tqdm(test_keys, desc=f"  Evaluating", leave=False):
                video_name = key.split('/')[-1]
                video_file = video_h5[video_name]
                
                # Load video features
                video = torch.from_numpy(video_file['features'][...].astype(np.float32)).unsqueeze(0).to(device)
                
                # Load text features  
                text = torch.from_numpy(text_feature_dict[video_name]).to(torch.float32).unsqueeze(0).to(device)
                
                # Load metadata
                n_frames = int(video_file['n_frames'][...])
                picks = video_file['picks'][...].astype(np.int32)
                change_points = video_file['change_points'][...].astype(np.int32)
                n_frame_per_seg = video_file['n_frame_per_seg'][...].astype(np.int32)
                
                # ==== KEY FIX: Load CONTINUOUS gtscore (NOT binary user_summary) ====
                gtscore = video_file['gtscore'][...].astype(np.float32)  # [T] continuous
                
                # Create masks
                num_frame = video.shape[1]
                num_sentence = text.shape[1]
                mask_video = torch.ones(1, num_frame, dtype=torch.long).to(device)
                mask_text = torch.ones(1, num_sentence, dtype=torch.long).to(device)
                
                video_cls_label = torch.zeros(1, num_frame).to(device)
                text_label = torch.zeros(1, num_sentence).to(device)
                
                # Create alignment masks
                import math
                frame_sentence_ratio = int(math.ceil(num_frame / num_sentence))
                video_to_text_mask = torch.zeros((num_frame, num_sentence), dtype=torch.long)
                text_to_video_mask = torch.zeros((num_sentence, num_frame), dtype=torch.long)
                for j in range(num_sentence):
                    start_frame = j * frame_sentence_ratio
                    end_frame = min((j + 1) * frame_sentence_ratio, num_frame)
                    video_to_text_mask[start_frame:end_frame, j] = 1
                    text_to_video_mask[j, start_frame:end_frame] = 1
                
                video_to_text_mask_list = [video_to_text_mask.to(device)]
                text_to_video_mask_list = [text_to_video_mask.to(device)]

                # Forward pass
                outputs = model(video=video, text=text, mask_video=mask_video, mask_text=mask_text,
                              video_label=video_cls_label, text_label=text_label,
                              video_to_text_mask_list=video_to_text_mask_list,
                              text_to_video_mask_list=text_to_video_mask_list)
                
                # Get predicted scores
                pred_scores = outputs['video_pred_cls'].sigmoid().cpu().numpy()[0]  # [T] at subsampled rate
                
                # ==== Compute rank correlation directly on subsampled (T) level ====
                # Ensure lengths match
                min_len = min(len(pred_scores), len(gtscore))
                pred_s = pred_scores[:min_len]
                gt_s = gtscore[:min_len]
                
                # Skip if no variance
                if np.std(pred_s) == 0 or np.std(gt_s) == 0:
                    continue
                    
                rho, _ = spearmanr(pred_s, gt_s)
                tau, _ = kendalltau(pred_s, gt_s)
                
                if not np.isnan(rho): 
                    all_spearman.append(rho)
                if not np.isnan(tau): 
                    all_kendall.append(tau)

        # Split Average
        avg_rho = np.mean(all_spearman) if all_spearman else 0.0
        avg_tau = np.mean(all_kendall) if all_kendall else 0.0
        
        print(f"  Split {split_idx} Results -> Rho: {avg_rho:.4f}, Tau: {avg_tau:.4f}")
        
        if all_spearman: split_spearman.append(avg_rho)
        if all_kendall: split_kendall.append(avg_tau)

    video_h5.close()
    
    # 3. Final Overall Average
    print("\n" + "="*60)
    print("FINAL RESULTS (5-Fold Cross Validation Average)")
    print("="*60)
    
    final_rho = np.mean(split_spearman) if split_spearman else 0.0
    final_tau = np.mean(split_kendall) if split_kendall else 0.0
    
    print(f"Average Spearman's rho: {final_rho:.4f}")
    print(f"Average Kendall's tau:  {final_tau:.4f}")
    print(f"Evaluated {len(split_spearman)} splits.")
    print("="*60)
    
    return final_rho, final_tau

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TVSum Rank Correlation Evaluation (Fixed)')
    parser.add_argument('--dataset', type=str, default='TVSum')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to model directory containing checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None, help='Specific checkpoint path')
    
    args, unknown = parser.parse_known_args()
    
    sys.argv = [sys.argv[0], '--dataset', args.dataset] 
    
    full_args = build_args()
    full_args.model_dir = args.model_dir
    full_args.checkpoint = args.checkpoint
    
    full_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    evaluate_rank_correlation(full_args)
