import h5py

h5_path = 'data/TVSum/feature/eccv16_dataset_tvsum_google_pool5.h5'
try:
    with h5py.File(h5_path, 'r') as f:
        video_name = list(f.keys())[0]
        v = f[video_name]
        print(f"Video: {video_name}")
        print(f"n_frames: {int(v['n_frames'][...])}")
        print(f"len(picks): {len(v['picks'][...])}")
        print(f"len(gtscore): {len(v['gtscore'][...])}")
except Exception as e:
    print(f"Error: {e}")
