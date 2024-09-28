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
            if score >= 0.7 and face_w > 10 and face_h > 10: 
                resize_ratio = 336 / max(hw[0], hw[1])
                resize_face_w = face_w * resize_ratio
                resize_face_h = face_h * resize_ratio
                if resize_face_w > 16 and resize_face_h > 16:
                    new_result.append(result)


        if len(new_result) < face_num_min_threshold or len(new_result) > face_num_max_threshold: #image requirement
            continue

        # 人脸的数量要完全对齐
        if len(new_result) < total_face:
            continue
        
        # # 介于 3/4 与 4/5 之间
        # if len(new_result) < total_face * 0.9:
        #     continue

        img_dict['result'] = new_result
        new_data.append(img_dict)
        num_imgs_face[len(new_result)-face_num_min_threshold] += 1
        new_data_face[len(new_result)-face_num_min_threshold].append(img_dict)

    # # save all data
    # new_json_path = json_path.replace('_new.json', '_clean.json')
    # with open(new_json_path, 'w') as f:
    #     json.dump(new_data, f, indent=4)

    # save each face number data

    save_dir = os.path.join(f"{os.path.dirname(json_path)}_refine", os.path.basename(json_path).replace('.json', '_clean'))   
    os.makedirs(save_dir, exist_ok=True)
    num_imgs_5_more_face = sum(num_imgs_face[5-face_num_min_threshold:])
    for i in range(face_hist_nums):
        new_json_path = os.path.join(save_dir, f'faces_{i+face_num_min_threshold}.json')
        with open(new_json_path, 'w') as f:
            json.dump(new_data_face[i], f, indent=4)
        
        print(i+face_num_min_threshold, num_imgs_face[i])     
          
    print(os.path.basename(json_path), len(data), len(new_data))

    print('num 5+ face:', num_imgs_5_more_face)
    return  num_imgs_5_more_face


root = 'face_detect_json_anno/faceCaption5M'

json_files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.json')]
print(json_files)
all_num_imgs_5_more_face = 0
for json_file in json_files:
    all_num_imgs_5_more_face+= clean_facecaption5m_json(json_file)
    # exit()
print('all_num_imgs_5_more_face:', all_num_imgs_5_more_face)