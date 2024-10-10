import os 
import json
import numpy as np
import random

dataset = 'laion_face_refine'
face_num_min_threshold = 1
face_num_max_threshold = 20
face_hist_nums = face_num_max_threshold - face_num_min_threshold + 1

root = f'face_detect_json_anno/{dataset}'

split_paths = [os.path.join(root, f) for f in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, f))]

resampled_data = []
face_num_total_dict = {"1":9094141, "2":1912845, "3":570578, "4":299364, "5":141890} # need change ratio

for split_path in split_paths:
    json_files = [os.path.join(split_path, f) for f in os.listdir(split_path) if f.endswith('.json')]

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        face_num = int(json_file.split('_')[-1].split('.')[0])
        if 6 <= face_num <= 20: # use all data
            resampled_data.extend(data)
            
        # use 30k data *4 
        elif 2 <= face_num <= 5:
            sample_prob =  120000 / face_num_total_dict[str(face_num)]
            for d in data:
                if random.random() < sample_prob:
                    resampled_data.append(d)
        
        # 50k
        elif face_num == 1:
            sample_prob = 200000 / face_num_total_dict[str(face_num)]
            for d in data:
                if random.random() < sample_prob:
                    resampled_data.append(d)
        else:
            raise ValueError(f'face_num: {face_num}')

print("total_data:", len(resampled_data))

new_json_path = os.path.join('face_detect_json_anno', f'laion_face_anno_json', f'{dataset}_828k.json')
os.makedirs(os.path.dirname(new_json_path), exist_ok=True)
with open(new_json_path, 'w') as f:
    json.dump(resampled_data, f, indent=4)