import os
import json
import numpy as np

split_num = 24

data_path = "face_detect_json_anno/faceCaption5M_anno_json/faceCaption5M_refine_216k.json"

dir_name = os.path.join(os.path.dirname(data_path), os.path.basename(data_path).replace('.json', '_split'))
os.makedirs(dir_name, exist_ok=True)

data = json.load(open(data_path, 'r'))
num_data = len(data)

split_size = num_data // split_num

sum_num = 0
data_list = []
for i in range(split_num):
    start = i * split_size
    end = (i + 1) * split_size
    if i == split_num - 1:
        end = num_data

    split_data = data[start:end]
    data_list.extend(split_data)
    sum_num += len(split_data)
    with open(os.path.join(dir_name, '{}.json'.format(i)), 'w') as f:
        json.dump(split_data, f, indent=4)

assert sum_num == num_data, "Split data error, sum_num: {}, num_data: {}".format(sum_num, num_data)

random_indexs = np.random.choice(num_data, 200, replace=False)
for i in random_indexs:
    assert data[i] == data_list[i], "Data error, i: {}".format(i)
