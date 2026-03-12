
import numpy as np
import torch
import os

try:
    path = 'data/TVSum/feature/text_roberta.npy'
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True).item()
        keys = list(data.keys())
        first_key = keys[0]
        feat = data[first_key]
        print(f"Video: {first_key}")
        print(f"Feature shape: {feat.shape}")
        
        # Check video feature to compare length
        vid_path = 'data/TVSum/feature/eccv16_dataset_tvsum_google_pool5.h5'
        import h5py
        if os.path.exists(vid_path):
            with h5py.File(vid_path, 'r') as f:
                vid_name = first_key.split('/')[-1]
                if vid_name in f:
                    vid_feat = f[vid_name]['features'][...]
                    print(f"Video feature shape: {vid_feat.shape}")
    else:
        print(f"Path not found: {path}")

except Exception as e:
    print(f"Error: {e}")
