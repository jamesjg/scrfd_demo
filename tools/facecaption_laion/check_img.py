import os
import json
from PIL import Image, ImageDraw
import cv2
from tools.info_wechat import sc_send

read_root = '/mnt/data/lanxing/Selected_Laion'

json_file = os.path.join('face_detect_json_anno/selectedLaionface/laion_face_refine_828k.json')
with open(json_file, 'r') as f:
    data = json.load(f)

no_find = []
no_cv2 = []
no_pil = []

for data_i in data:
    file_name = data_i["path"]
    if not os.path.exists(os.path.join(read_root, file_name)):
        no_find.append(file_name)
        continue

    try:
        img = Image.open(os.path.join(read_root,file_name))
        assert img is not None
        assert img.size[0] > 0 and img.size[1] > 0
    except Exception as e:
        np_pil.append(file_name)
        print(file_name, e)
        continue
    
    try:
        img = cv2.imread(os.path.join(read_root,file_name))
        assert img is not None
        assert img.shape[0] > 0 and img.shape[1] > 0
    except Exception as e:
        no_cv2.append(file_name)
        print(file_name, e)

print(f"check image done, no find:{len(no_find)}, no cv2:{len(no_cv2)}, no pil:{len(no_pil)}", f"no find:{no_find}, no cv2:{no_cv2}, no pil:{no_pil}")
sc.send(f"check image done, no find:{len(no_find)}, no cv2:{len(no_cv2)}, no pil:{len(no_pil)}", f"no find:{no_find}, no cv2:{no_cv2}, no pil:{no_pil}")

print("check image done")