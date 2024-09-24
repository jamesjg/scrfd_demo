import os
import json


cmd = ""


gpu_ids = [2,3,4,5,6,7]
data_root = "/mnt/data/lanxing/faceCaption5M/images/images"
dirs = [dir for dir in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, dir))]
img_folders = [os.path.join(data_root, dir) for dir in dirs]
dataset = "faceCaption5M"


for i, img_folder in enumerate(img_folders):
    output_json_path =f'face_detect_json_anno/{dataset}/scrfd_{os.path.basename(img_folder)}.json'
    os.makedirs(f'face_detect_json_anno/{dataset}/', exist_ok=True)
    cmd += f"CUDA_VISIBLE_DEVICES={gpu_ids[i%len(gpu_ids)]}  python tools/scrfd_facecaption5m.py --folder_path {img_folder} --output_json_path {output_json_path} & "
    if i == len(img_folders)-1:
        cmd = cmd[:-2]

print(cmd)
os.system(cmd)