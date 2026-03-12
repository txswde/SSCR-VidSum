
import h5py
import numpy as np
import sys

def analyze_tvsum_gt():
    h5_path = 'data/TVSum/feature/eccv16_dataset_tvsum_google_pool5.h5'
    print(f"Analyzing {h5_path}...")
    
    with h5py.File(h5_path, 'r') as f:
        videos = list(f.keys())
        print(f"Total videos: {len(videos)}")
        
        all_means = []
        all_stds = []
        all_zeros = []
        
        for vid in videos:
            gtscore = f[vid]['gtscore'][...]
            
            # Print first video stats for example
            if len(all_means) == 0:
                print(f"\nExample Video: {vid}")
                print(f"gtscore shape: {gtscore.shape}")
                print(f"gtscore range: [{gtscore.min():.4f}, {gtscore.max():.4f}]")
                print(f"gtscore mean: {gtscore.mean():.4f}")
            
            all_means.append(gtscore.mean())
            all_stds.append(gtscore.std())
            all_zeros.append((gtscore == 0).mean())
            
        print(f"\nOverall Statistics:")
        print(f"Avg GT Score Mean: {np.mean(all_means):.4f}")
        print(f"Avg GT Score Std: {np.mean(all_stds):.4f}")
        print(f"Avg Zero Ratio: {np.mean(all_zeros):.4f}")

if __name__ == "__main__":
    analyze_tvsum_gt()
