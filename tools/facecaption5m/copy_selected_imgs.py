import os
import json
from PIL import Image, ImageDraw
import cv2

read_root = '/mnt/data/lanxing/faceCaption5M/images/images'
write_root = '/mnt/data/lanxing/faceCaption5M/images/select_images'

json_file = os.path.join('face_detect_json_anno/faceCaption5M_anno_json/faceCaption5M_refine_216k.json')
with open(json_file, 'r') as f:
    data = json.load(f)

for data_i in data:
    file_name = os.path.join(read_root,data_i["path"])
    assert os.path.exists(file_name), f"{file_name} not exists"
    write_file_name = os.path.join(write_root,data_i["path"])
    os.makedirs(os.path.dirname(write_file_name), exist_ok=True)
    os.system(f"cp {file_name} {write_file_name}")

print("convert data path done")

# for data_i in data:
#     file_name = data_i["path"]
#     assert os.path.exists(os.path.join(write_root, file_name))
#     try:
#         img = Image.open(os.path.join(write_root,file_name))
#         assert img is not None
#         assert img.size[0] > 0 and img.size[1] > 0
#     except Exception as e:
#         print(file_name, e)
    
#     try:
#         img = cv2.imread(os.path.join("/mnt/data/lanxing/faceCaption5M/images/images",file_name))
#         assert img is not None
#         assert img.shape[0] > 0 and img.shape[1] > 0
#     except Exception as e:
#         print(file_name, e)

    # imgdraw = ImageDraw.Draw(img)
    # faces = data_i["result"]
    # for face in faces:
    #     x,y,w,h = face["facial_area"]
    #     x = float(x)
    #     y = float(y)
    #     w = float(w)
    #     h = float(h)
    #     imgdraw.rectangle([x,y,x+w,y+h], outline="red")
    
    # img.save(os.path.join("vis_219k_samples", os.path.basename(data_i["path"])))

print("check image done")