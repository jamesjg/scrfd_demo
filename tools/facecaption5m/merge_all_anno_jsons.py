import json
import os


def merge_all_anno_jsons(json_path, output_path, anno_type_names, json_index=0):
    output_json_path = os.path.join(output_path, f'{json_index}.json')
    os.makedirs(output_path, exist_ok=True)

    merged_json = []

    json_data_list = []
    for anno_type_name in anno_type_names:
        json_file = [os.path.join(json_path, anno_type_name, f) for f in os.listdir(os.path.join(json_path, anno_type_name)) if f.startswith(f'{json_index}_')]
        assert len(json_file) == 1
        json_file = json_file[0]
        
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            
        print(f'{os.path.basename(json_file)} loaded')

        json_data_list.append(json_data)

    
    #check json_data_list
    for i in range(len(json_data_list)):
        assert len(json_data_list[i]) == len(json_data_list[0]) # number
        for data_i in range(len(json_data_list[i])):
            assert json_data_list[i][data_i]['path'] == json_data_list[0][data_i]['path']
            assert json_data_list[i][data_i]['hw'] == json_data_list[0][data_i]['hw']
            for face_i in range(len(json_data_list[i][data_i]['result'])):
                assert json_data_list[i][data_i]['result'][face_i]['score'] == json_data_list[0][data_i]['result'][face_i]['score']
                assert json_data_list[i][data_i]['result'][face_i]['facial_area'] == json_data_list[0][data_i]['result'][face_i]['facial_area']
                assert json_data_list[i][data_i]['result'][face_i]['landmarks'] == json_data_list[0][data_i]['result'][face_i]['landmarks']
        print(f'{anno_type_names[i]} check done')

    # merge json_data_list
    for data_i in range(len(json_data_list[0])):
        merged_data_i_dict = {}
        merged_data_i_dict['path'] = json_data_list[0][data_i]['path']
        merged_data_i_dict['hw'] = json_data_list[0][data_i]['hw']
        merged_data_i_dict['result'] = []
        for face_i in range(len(json_data_list[0][data_i]['result'])):
            merged_face_i_dict = {}
            merged_face_i_dict['score'] = json_data_list[0][data_i]['result'][face_i]['score']
            merged_face_i_dict['facial_area'] = json_data_list[0][data_i]['result'][face_i]['facial_area']
            merged_face_i_dict['landmarks'] = json_data_list[0][data_i]['result'][face_i]['landmarks']
            
            # age_gender
            merged_face_i_dict['age'] = json_data_list[0][data_i]['result'][face_i]['age']
            merged_face_i_dict['gender'] = json_data_list[0][data_i]['result'][face_i]['gender']
            
            # attr
            merged_face_i_dict['attr'] = json_data_list[1][data_i]['result'][face_i]['Attr_Farl']

            # exp
            merged_face_i_dict['exp'] = json_data_list[2][data_i]['result'][face_i]['exp']

            # headpose
            merged_face_i_dict['yaw_pitch_roll'] = json_data_list[3][data_i]['result'][face_i]['yaw_pitch_roll']

            merged_data_i_dict['result'].append(merged_face_i_dict)

        merged_json.append(merged_data_i_dict)
    
    with open(output_json_path, 'w') as f:
        json.dump(merged_json, f, indent=4)

if __name__ == '__main__':
    json_path = 'face_detect_json_anno/Anno_jsons'
    output_path = 'face_detect_json_anno/Anno_jsons/All_info'
    anno_type_names = ['Age_gender', 'Attr', 'Exp', 'Headpose']
    merge_all_anno_jsons(json_path, output_path, anno_type_names, 0)

