import json 
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def cos(x):
    x = np.deg2rad(x)
    return np.cos(x)

def sin(x):
    x = np.deg2rad(x)
    return np.sin(x)


def visualize_all_anno(img_folder, json_path, output_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    for data_i in range(len(json_data)):
        img_path = json_data[data_i]['path']
        img_path = os.path.join(img_folder, img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        for face_i in range(len(json_data[data_i]['result'])):
            facial_area = json_data[data_i]['result'][face_i]['facial_area']
            facial_area = [int(eval(x)) for x in facial_area]
            landmarks = json_data[data_i]['result'][face_i]['landmarks']
            landmarks = {key: [int(eval(x)) for x in value] for key, value in landmarks.items()}
            x_min, y_min, w, h = facial_area
            x_max, y_max = x_min + w, y_min + h
            age = json_data[data_i]['result'][face_i]['age']
            gender = json_data[data_i]['result'][face_i]['gender']
            attr = json_data[data_i]['result'][face_i]['attr']
            exp = json_data[data_i]['result'][face_i]['exp']
            yaw_pitch_roll = json_data[data_i]['result'][face_i]['yaw_pitch_roll']

            # draw facial area
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            font_scale = w / 200
            font_thickness = max(1, int(w / 200))
            font = cv2.FONT_HERSHEY_SIMPLEX


            

            # draw attr
            for j, (attribute, prob) in enumerate(attr):
                text = f"{attribute}: {prob:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                cv2.rectangle(img, (x_min, y_min - text_height - 10 - j * (text_height + 5)), (x_min + text_width, y_min - 10 - j * (text_height + 5)), (0, 255, 255), -1)
                cv2.putText(img, text, (x_min, y_min - 10 - j * (text_height + 5)), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

            # draw exp
            expression = exp[0]
            prob = exp[1]
            text = f"{expression}: {prob:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            cv2.rectangle(img, (x_min, y_min - text_height - 10 - len(attr) * (text_height + 5)), (x_min + text_width, y_min - 10 - len(attr) * (text_height + 5)), (0, 255, 0), -1)
            cv2.putText(img, text, (x_min, y_min - 10 - len(attr) * (text_height + 5)), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

            # draw age and gender
            text = f"age: {age:.2f}, gender: {gender}"
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            cv2.rectangle(img, (x_min, y_max + 10), (x_min + text_width, y_max + 10 + text_height), (255, 0, 255), -1)
            cv2.putText(img, text, (x_min, y_max + 10 + text_height), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

            # draw yaw_pitch_roll
            size = 0.5 * w
            tdx = landmarks['nose'][0]
            tdy = landmarks['nose'][1]
            yaw, pitch, roll = yaw_pitch_roll
            x1 = size * (cos(yaw) * cos(roll)) + tdx
            y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(yaw) * sin(pitch)) + tdy

            x2 = size * (-cos(yaw) * sin(roll)) + tdx
            y2 = size * (cos(pitch) * cos(roll) - sin(yaw) * sin(pitch) * sin(roll)) + tdy

            x3 = size * (sin(yaw)) + tdx
            y3 = size * (-cos(yaw) * sin(pitch)) + tdy

            cv2.line(img, (tdx, tdy), (int(x1), int(y1)), (0, 0, 255), 4)
            cv2.line(img, (tdx, tdy), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.line(img, (tdx, tdy), (int(x3), int(y3)), (255, 0, 0), 4)

        os.makedirs(output_path, exist_ok=True)
        cv2.imwrite(os.path.join(output_path, f"anno_{os.path.basename(img_path)}"), img)



if __name__ == "__main__":
    img_folder = "/mnt/data/lanxing/faceCaption5M/images/images/"
    json_index = 0
    json_path = f'face_detect_json_anno/Anno_jsons/All_info/{json_index}.json'
    output_path = f"vis_results/vis_anno/vis_attributes_json{json_index}"
    visualize_all_anno(img_folder, json_path, output_path)