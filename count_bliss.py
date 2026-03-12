import json
import os

for split in ['train', 'val', 'test']:
    path = f'data/BLiSS/annotation/{split}.json'
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
                num_clips = len(data)
                num_videos = len(set(v['video_id'] for v in data.values()))
                print(f"{split} split: {num_clips} clips, {num_videos} videos")
        except Exception as e:
            print(f"Error loading {split}: {e}")
