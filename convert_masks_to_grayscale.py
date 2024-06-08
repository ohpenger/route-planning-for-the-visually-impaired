import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import sys

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def convert_color_mapping(labels,color_mapping):
    mapping_from_color2trainid = {}
    for item in color_mapping["labels"]:
        readable = item["readable"].lower()
        if readable in labels:
            color = item["color"]
            mapping_from_color2trainid[tuple(color)]=labels[readable]
    return mapping_from_color2trainid


def rgb_to_object_id(rgb_value, mapping_from_color2trainid):
    if tuple(rgb_value) in mapping_from_color2trainid:
        return mapping_from_color2trainid[tuple(rgb_value)]
    else:
        return 255


def convert_masks(input_folder, output_folder, mapping_from_color2trainid):
    os.makedirs(output_folder, exist_ok=True)

    for filename in tqdm(os.listdir(input_folder), desc="Processing files", unit="file"):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            mask = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert RGB values to object IDs
            mask_object_ids = np.zeros_like(mask[:, :, 0], dtype=np.uint8)

            for rgb_value, obj_id in mapping_from_color2trainid.items():
                rgb_value = list(rgb_value)
                mask_object_ids[np.all(mask == rgb_value, axis=-1)] = obj_id

            # Save the grayscale image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, mask_object_ids)


if __name__ == "__main__":
    labels = {"unlabeled": 0,
    	      "road": 7,
              "sidewalk":8 ,
              "traffic light": 19,
              "traffic sign": 20,
              "person": 24,
              "car": 26,
              "bicycle":33}

    json_path = sys.argv[1] #"F:\data\cityscapesScripts-master\Mapillary-Vistas-1000-sidewalks\config.json"
    color_mapping = load_json(json_path)

    mapping_from_color2trainid = convert_color_mapping(labels,color_mapping)


    training_folder = sys.argv[2] #"path/to/your/training/images/folder"
    training_output_folder = sys.argv[3] #"path/to/your/training/output/folder"
    convert_masks(training_folder, training_output_folder, mapping_from_color2trainid)

    validation_folder = sys.argv[4] #"path/to/your/training/images/folder"
    validation_output_folder = sys.argv[5] #"path/to/your/training/images/folder"
    convert_masks(validation_folder, validation_output_folder, mapping_from_color2trainid)

    testing_folder = sys.argv[6] #"path/to/your/training/images/folder"
    testing_output_folder = sys.argv[7] #"path/to/your/training/images/folder"
    convert_masks(testing_folder, testing_output_folder, mapping_from_color2trainid)

