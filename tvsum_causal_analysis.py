"""
TVSum Causal Analysis & Visualization
======================================
Visualizes the learned causal alignment in TVSum and compares it with 
the naive linear alignment heuristic.

Since TVSum lacks ground-truth sentence-level timestamps, we cannot compute
GT correlation. Instead, we analyze:
1. Deviation from Diagonal (Linear Heuristic).
2. Sparsity / Focus.
3. Qualitative visualization of selected samples.
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import build_args
from models import Model_VideoSumm
from datasets import VideoSummDataset
from causal_alignment_discovery import CausalAlignmentDiscovery

def analyze_tvsum_alignment(args):
    print("\n" + "="*60)
    print("TVSum Causal Alignment Analysis")
    print("="*60)

    # Force enable alignment discovery for analysis
    args.enable_causal_alignment = True
    
    # Load model
    model = Model_VideoSumm(args=args)
    if not args.model_dir:
        # Default to the innovation experiment dir
        args.model_dir = "logs/TVSum_adaptive_causal"
        
    # Try default path
    checkpoint_path = os.path.join(args.model_dir, 'checkpoint', 'model_best_split0.pt')
    
    # Try nested TVSum path (common if train.py appends dataset name)
    if not os.path.exists(checkpoint_path):
         checkpoint_path_nested = os.path.join(args.model_dir, 'TVSum', 'checkpoint', 'model_best_split0.pt')
         if os.path.exists(checkpoint_path_nested):
             checkpoint_path = checkpoint_path_nested
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        return None

    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    # Load dataset (Split 0 Test Set)
    # Note: We need the split keys. For simplicity, we'll just load the full dataset 
    # or rely on the split file if available.
    # Assuming 'data/TVSum/splits.json' or similar exists, but lets try to just load 
    # the dataset using the standard logic or just keys from the checkpoint if saved?
    # Train/Test splits are usually defined in valid_splits.json or similar.
    # For visualization, we can just look at a few examples from the full set.
    
    # Quick hack: Load all keys from the dataset file
    import h5py
    h5_path = f'{args.data_root}/{args.dataset}/feature/eccv16_dataset_tvsum_google_pool5.h5'
    with h5py.File(h5_path, 'r') as f:
        all_keys = list(f.keys())
    
    # Use first 5 videos for visualization
    test_keys = all_keys[:5]
    print(f"Analyzing {len(test_keys)} samples...")
    
    dataset = VideoSummDataset(keys=test_keys, args=args)
    
    # Discovery Module
    discovery = CausalAlignmentDiscovery(temperature=1.0)
    
    save_dir = os.path.join("interpretability_results", "TVSum_alignment")
    os.makedirs(save_dir, exist_ok=True)
    
    results = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        # Unpack based on VideoSummDataset.__getitem__
        # It's a bit long, let's just grab the first few items we need
        video = sample[0].unsqueeze(0).to(args.device)
        text = sample[1].unsqueeze(0).to(args.device)
        mask_video = sample[2].unsqueeze(0).to(args.device)
        mask_text = sample[3].unsqueeze(0).to(args.device)
        
        # Labels (needed for forward pass even if not used)
        video_label = sample[4].unsqueeze(0).to(args.device)
        # Create dummy text_label to avoid NoneType error in select_contrastive_embedding
        # Shape should match [B, T_t] with zeros (no key sentences marked)
        T_t = text.shape[1]
        text_label = torch.zeros(1, T_t, dtype=torch.float32).to(args.device)
        # Let's create a minimal forward pass wrapper or just provide all
        
        # Simplified: We just want to run Discovery.
        # Discovery needs 'model' and inputs.
        
        # We need to construct the args for discovery.forward
        # But discovery expects a model forward call internally? 
        # Actually causal_alignment_discovery.forward calls (model, video, text...)
        
        # We need to reconstruct the full batch input as expected by model forward
        # Let's mock the necessary parts
        
        # Actually, let's manually run the intervention loop here for simplicity and control
        # instead of relying on CausalAlignmentDiscovery.forward which might be coupled to training args
        
        T_v = video.shape[1]
        T_t = text.shape[1]
        
        print(f"Sample {i}: Video {T_v} frames, Text {T_t} sentences")
        
        # 1. Original Prediction
        with torch.no_grad():
             # We need to construct mask lists as model expects
            video_to_text_mask_list = [sample[-4].to(args.device)] # 4th from end in dataset
            text_to_video_mask_list = [sample[-3].to(args.device)]
            
            # Disable linear mask for prediction to get "natural" behavior
            # (Though 'adaptive_causal' model was trained with ones, so we pass ones)
            # But wait, we want to see what the model *learned*.
            # The model learned to attend based on content.
            
            # To measure "Intervention Dissimilarity", we do the intervention loop.
            
            # Get base prob
            # Note: Model_VideoSumm returns a dict
            out_orig = model(video=video, text=text, mask_video=mask_video, mask_text=mask_text, 
                           video_label=video_label, text_label=text_label,
                           video_to_text_mask_list=video_to_text_mask_list, 
                           text_to_video_mask_list=text_to_video_mask_list)
            
            prob_orig = torch.sigmoid(out_orig['video_pred_cls']) # [B, T]
            
            # 2. Intervention Loop
            heatmap = torch.zeros(T_v, T_t)
            
            for j in range(T_t):
                text_masked = text.clone()
                text_masked[:, j, :] = 0
                
                out_masked = model(video=video, text=text_masked, mask_video=mask_video, mask_text=mask_text,
                                 video_label=video_label, text_label=text_label,
                                 video_to_text_mask_list=video_to_text_mask_list,
                                 text_to_video_mask_list=text_to_video_mask_list)
                prob_masked = torch.sigmoid(out_masked['video_pred_cls'])
                
                diff = torch.abs(prob_orig - prob_masked).squeeze() # [T_v]
                heatmap[:, j] = diff.cpu()
        
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
            
        # === Quantitative Analysis ===
        # 1. Diagonal Concentration: How much mass is near the linear diagonal?
        ratio = T_v / T_t
        diag_mask = torch.zeros_like(heatmap)
        band_width = 0.15 * T_v  # 15% tolerance window
        
        for t_idx in range(T_t):
            center = t_idx * ratio + ratio/2
            start = max(0, int(center - band_width))
            end = min(T_v, int(center + band_width))
            diag_mask[start:end, t_idx] = 1.0
            
        diag_score = (heatmap * diag_mask).sum() / (heatmap.sum() + 1e-8)
        
        # 2. Sparsity (Gini Coefficient): How focused is the attention?
        flat_map = heatmap.flatten()
        sorted_map, _ = torch.sort(flat_map)
        n = len(flat_map)
        index = torch.arange(1, n + 1, device=heatmap.device).float()
        gini = (2 * torch.sum(index * sorted_map) / (n * torch.sum(sorted_map) + 1e-8)) - (n + 1) / n
        
        print(f"  - Diagonal Concentration: {diag_score:.2%}")
        print(f"  - Sparsity (Gini): {gini:.4f}")
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(heatmap.numpy(), aspect='auto', cmap='hot', interpolation='nearest')
        
        # Plot Linear Alignment as overlay (Diagonal)
        # Linear: frame i corresponds to sentence floor(i / ratio)
        ratio = T_v / T_t
        x = np.arange(T_t)
        y = x * ratio + ratio/2
        ax.plot(x, y, 'g--', linewidth=2, label='Linear Heuristic')
        
        ax.set_title(f"Learned Causal Alignment (Sample {i})\nGreen Line = Linear Assumption")
        ax.set_xlabel("Sentence Index")
        ax.set_ylabel("Frame Index")
        plt.colorbar(im)
        plt.legend()
        
        save_path = os.path.join(save_dir, f"sample_{i}_alignment.png")
        plt.savefig(save_path)
        plt.close()
        results.append(save_path)
        
    print(f"\nSaved {len(results)} visualizations to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TVSum')
    parser.add_argument('--model_dir', type=str, default=None)
    
    # Add other necessary args from config
    args = parser.parse_args()
    
    # Hack to get full config
    sys.argv = [sys.argv[0], '--dataset', 'TVSum']
    if parser.parse_args().model_dir:
        sys.argv.extend(['--model_dir', parser.parse_args().model_dir])
        
    full_args = build_args()
    # Override
    full_args.model_dir = args.model_dir 
    full_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    analyze_tvsum_alignment(full_args)
