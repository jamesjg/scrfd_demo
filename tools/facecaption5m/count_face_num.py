import json
import os
import numpy as np

face_num_min_threshold = 1
face_num_max_threshold = 20
face_hist_nums = face_num_max_threshold - face_num_min_threshold + 1
json_file = 'face_detect_json_anno/faceCaption5M_219k/faceCaption5M_219k.json'

total = 0

total_num_imgs_face = [0] * face_hist_nums

with open(json_file, 'r') as f:
    data = json.load(f)
for data_i in data:
    face_num = len(data_i["result"])
    total_num_imgs_face[face_num - face_num_min_threshold] += 1

for i, num in enumerate(total_num_imgs_face):
    print(f"{i + face_num_min_threshold}: {num}")

  