import json
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw

json_file = 'face_detect_json_anno/faceCaption5M_anno_json/faceCaption5M_refine_216k.json'

with open(json_file, 'r') as f:
    data = json.load(f)

random_indexs = np.random.choice(len(data), 50, replace=False)

for i in random_indexs:
    data_i = data[i]
    file_name = data_i["path"].split('/')[-1]
    split_folder = os.path.join(data_i["path"].split('/')[0], file_name[:5])
    new_file_name = os.path.join(split_folder, file_name)
    # new_file_name = data_i["path"]
    img = Image.open(os.path.join("/mnt/data/lanxing/faceCaption5M/images/images",new_file_name))
    imgdraw = ImageDraw.Draw(img)
    faces = data_i["result"]
    for face in faces:
        x,y,w,h = face["facial_area"]
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)
        imgdraw.rectangle([x,y,x+w,y+h], outline="red")
    
    img.save(os.path.join("vis_samples", os.path.basename(data_i["path"])))


  