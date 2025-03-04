import os
import cv2
import numpy as np
import json
import pandas as pd
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# Function to process a single image
def process_image(image_path, class_names, output_dir, predictor):
    print(f"Processing: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return []

    print("Image successfully loaded")

    try:
        print("Performing inference")
        outputs = predictor(image)
        print("Inference completed")
    except Exception as e:
        print(f"Error in image inference {image_path}: {e}")
        return []

    instances = outputs["instances"].to("cpu")
    pred_boxes = instances.pred_boxes
    pred_classes = instances.pred_classes
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    records = []  # List to accumulate records

    for i in range(len(pred_classes)):
        box = pred_boxes[i].tensor.numpy()[0]
        cls = pred_classes[i].item()
        x1, y1, x2, y2 = map(int, box)
        cropped_image = image[y1:y2, x1:x2]
        class_name = class_names[cls]
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Save cropped images as PNG
        output_image_path = os.path.join(class_dir, f"{base_name}_extracted_{i}.png")
        cv2.imwrite(output_image_path, cropped_image)
        print(f"Extracted image saved at: {output_image_path}")

        year = base_name.split('_')[0]
        page = base_name.split('_')[1]
        width, height = image.shape[1], image.shape[0]
        crop_width, crop_height = x2 - x1, y2 - y1
        proportion = (crop_width * crop_height) / (width * height)

        records.append({
            "image_id": f"{base_name}_extracted_{i}",
            "image_path": output_image_path,
            "year": year,
            "page": page,
            "class": class_name,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "proportion": proportion
        })

        json_data = {
            "image_id": f"{base_name}_extracted_{i}",
            "cropped_coordinates": [{
                "class": class_name,
                "coordinates": [x1, y1, x2, y2]
            }]
        }

        json_output_path = os.path.join(class_dir, f"{base_name}_extracted_{i}.json")
        
        try:
            if os.path.exists(json_output_path):
                print(f"JSON file already exists: {json_output_path}. Skipping.")
                continue
            
            with open(json_output_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=4)
            print(f"Coordinates saved in JSON: {json_output_path}")
        except PermissionError:
            print(f"Error: Permission denied to write file {json_output_path}. Skipping.")
            continue

    return records  # Return records

# Main function to process multiple images in a directory
def process_images_in_directory(input_dir, class_names):
    output_dir = "E://The Delineator Cropped Images//1911"
    os.makedirs(output_dir, exist_ok=True)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = "C://Users//lumor//Downloads/model_final.pth"
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)

    predictor = DefaultPredictor(cfg)

    all_records = []  # List to accumulate all records

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(('.png', '.jpg')):  # Add other formats if necessary
                image_path = os.path.join(root, filename)
                records = process_image(image_path, class_names, output_dir, predictor)
                if records:
                    all_records.extend(records)

    # Create a DataFrame and save as CSV
    df = pd.DataFrame(all_records)
    csv_output_path = os.path.join(output_dir, "extracted_records_1911.csv")
    df.to_csv(csv_output_path, index=False)
    print(f"Records saved in CSV: {csv_output_path}")

# Path to the folder containing your images
input_directory = "E://The Delineator Images//1911//1911"
class_names = ["Illustration/Photograph", "Photograph", "Comics/Cartoon", "Editorial Cartoon", "Map", "Headline", "Ad"]

process_images_in_directory(input_directory, class_names)
