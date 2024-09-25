import json
import os

face_num_min_threshold = 1
face_num_max_threshold = 20
face_hist_nums = face_num_max_threshold - face_num_min_threshold + 1

def clean_facecaption5m_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    new_data = []

    num_imgs_face = [0 for _ in range(face_hist_nums)] 
    new_data_face = [[] for _ in range(face_hist_nums)]
    for img_dict in data:  # each image
        ori_result = img_dict['result']
        hw = img_dict['hw']
        if len(ori_result) < face_num_min_threshold or hw[0] < 100 or hw[1] < 100: # image requirement
            continue
        
        
        new_result = []
        total_face = len(ori_result)
        for result in ori_result: # each face
            score = float(result['score'])
            face_w, face_h = float(result['facial_area'][2]), float(result['facial_area'][3])
            # face requirement
            if score >= 0.75 and face_w > 10 and face_h > 10: 
                resize_ratio = 336 / max(hw[0], hw[1]) 
                resize_face_w = face_w * resize_ratio
                resize_face_h = face_h * resize_ratio
                if resize_face_w > 15 and resize_face_h > 15:
                    new_result.append(result)


        if len(new_result) < face_num_min_threshold or len(new_result) > face_num_max_threshold: #image requirement
            continue

        # 前五个人脸的数量要完全对齐
        if len(new_result) < 7 and len(new_result) < total_face:
            continue
        
        # 介于 3/4 与 4/5 之间
        if len(new_result) < total_face * 0.78:
            continue

        img_dict['result'] = new_result
        new_data.append(img_dict)
        num_imgs_face[len(new_result)-face_num_min_threshold] += 1
        new_data_face[len(new_result)-face_num_min_threshold].append(img_dict)

    # # save all data
    # new_json_path = json_path.replace('_new.json', '_clean.json')
    # with open(new_json_path, 'w') as f:
    #     json.dump(new_data, f, indent=4)

    # save each face number data

    save_dir = os.path.join(os.path.dirname(json_path), os.path.basename(json_path).replace('.json', '_clean'))   
    os.makedirs(save_dir, exist_ok=True)
    for i in range(face_hist_nums):
        new_json_path = os.path.join(save_dir, f'faces_{i+face_num_min_threshold}.json')
        with open(new_json_path, 'w') as f:
            json.dump(new_data_face[i], f, indent=4)
        
        print(i+face_num_min_threshold, num_imgs_face[i])     

    print(os.path.basename(json_path), len(data), len(new_data))


    

root = 'face_detect_json_anno/faceCaption5M'


json_files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.json')]
print(json_files)
for json_file in json_files:
    clean_facecaption5m_json(json_file)