import os
import json
from PIL import Image, ImageDraw
import cv2

root = 'face_detect_json_anno/faceCaption5M_anno_json'


json_file = os.path.join(root, 'faceCaption5M_refine_215k.json')
new_json_file = os.path.join(root, 'faceCaption5M_refine_215k_new.json')

with open(json_file, 'r') as f:
    data = json.load(f)
for data_i in data:
    file_name = data_i["path"].split('/')[-1]
    split_folder = os.path.join(data_i["path"].split('/')[0], file_name[:5])
    new_file_name = os.path.join(split_folder, file_name)
    data_i["path"] = new_file_name
    assert os.path.exists(os.path.join("/mnt/data/lanxing/faceCaption5M/images/images", new_file_name))
with open(new_json_file, 'w') as f:
    json.dump(data, f, indent=4)

print("convert data path done")

for data_i in data:
    file_name = data_i["path"]
    assert os.path.exists(os.path.join("/mnt/data/lanxing/faceCaption5M/images/images", file_name))
    try:
        img = Image.open(os.path.join("/mnt/data/lanxing/faceCaption5M/images/images",file_name))
        assert img is not None
        assert img.size[0] > 0 and img.size[1] > 0
    except Exception as e:
        print(file_name, e)
    
    try:
        img = cv2.imread(os.path.join("/mnt/data/lanxing/faceCaption5M/images/images",file_name))
        assert img is not None
        assert img.shape[0] > 0 and img.shape[1] > 0
    except Exception as e:
        print(file_name, e)
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