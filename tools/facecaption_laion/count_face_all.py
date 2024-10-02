import json
import os
import numpy as np
dataset = 'laion_face_refine'
face_num_min_threshold = 1
face_num_max_threshold = 20
face_hist_nums = face_num_max_threshold - face_num_min_threshold + 1
root = f'face_detect_json_anno/{dataset}'

# 做一个表，记录每个json文件中，每个人脸数的图片数量，以及总数
'''
    --face_detect_json_anno/laion_face
        --scrfd_split_00000_clean
            --faces_2.json
            --faces_3.json
            ...
        --scrfd_split_00001_clean
            --faces_2.json
            --faces_3.json
            ...

'''

'''     face2 face3
split 0
split 1

'''

total = 0


split_paths = [os.path.join(root, f) for f in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, f))]
num_table = np.zeros((len(split_paths)+1, face_hist_nums))

total_num_imgs_face = [0] * face_hist_nums
for split_path in split_paths:
    json_files = [os.path.join(split_path, f) for f in os.listdir(split_path) if f.endswith('.json')]

    for json_file in json_files:

        
        with open(json_file, 'r') as f:
            data = json.load(f)
        face_num_id = json_file.split('_')[-1].split('.')[0]
        face_num_id = int(face_num_id) - face_num_min_threshold

        total_num_imgs_face[face_num_id] += len(data)
        num_table[int(split_path.split('_')[-2]) - 1][face_num_id] = len(data)


    #     print(json_file, len(data))



print(total_num_imgs_face)

num_table[-1] = total_num_imgs_face
np.savetxt(f'face_detect_json_anno/{dataset}/num_table.txt', num_table, fmt='%d', delimiter=' ')


  