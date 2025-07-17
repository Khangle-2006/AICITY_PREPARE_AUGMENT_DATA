import os
import orjson
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import shutil

from config import TRAIN_DATA, TEST_DATA
from utils.gen_pseudo_labels import generate_pseudo_labels
from utils.filter_loaf import filter_loaf
from utils.merge_loaf_labels import merge_loaf_labels
from utils.make_night_images import generate_nighttime_images



def fast_copy(src, dst, buffer_size=16 * 1024 * 1024):
    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
        shutil.copyfileobj(fsrc, fdst, length=buffer_size)

def check_data_integrity(data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for dataset, info in data_dict.items():
        images_dir = info.get("images_dir")
        label_file = info.get("label")
        
        assert os.path.exists(images_dir), f"Images directory for {dataset} does not exist: {images_dir}"

        if label_file and not os.path.exists(label_file):
            print(f"Warning: Label file for {dataset} does not exist: {label_file}")

def generate_labels(args, DATA, OUTPUT_DIR):
    OUTPUT_LABEL = {}
    
    for dataset, info in DATA.items():
        images_dir = info.get("images_dir")
        label_file = info.get("label")
        output_label_file = os.path.join(OUTPUT_DIR, f'{dataset}_labels.json')
        print(f"Processing dataset: {dataset}")
        
        if not label_file or dataset == "LOAF":
            generate_pseudo_labels(
                config_file="config/CO-DETR/projects/CO-DETR/configs/codino/train_all.py",
                checkpoint_file=args.codetr,
                images_dir=images_dir,
                output_file=output_label_file,
                device=args.device,
            )
            
            if dataset == "LOAF":
                if label_file and os.path.exists(label_file):
                    merge_file = os.path.join(OUTPUT_DIR, f'{dataset}_merged_labels.json')
                    merge_loaf_labels(
                        pseudo_file=output_label_file,
                        real_file=label_file,
                        output_file=merge_file,
                        iou_threshold=0.2
                    )
                    
                    filtered_file = os.path.join(OUTPUT_DIR, f'{dataset}20k_labels.json')
                    filter_loaf(merge_file, filtered_file, top_k=20000)
                    output_label_file = filtered_file
                else:
                    filtered_file = os.path.join(OUTPUT_DIR, f'{dataset}20k_labels.json')
                    filter_loaf(output_label_file, filtered_file, top_k=20000)
                    output_label_file = filtered_file
                
        else:
            fast_copy(label_file, output_label_file)
            print(f"Skipping {dataset} as it has a label file: {label_file}")
            
        OUTPUT_LABEL[dataset] = output_label_file
        
    return OUTPUT_LABEL

def merge_all_labels(label_files, output_file):
    """Merge multiple COCO label files into one."""
    print(f"Merging {len(label_files)} label files into {output_file}")
    
    merged_coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "Bus"},
            {"id": 1, "name": "Bike"},
            {"id": 2, "name": "Car"},
            {"id": 3, "name": "Pedestrian"},
            {"id": 4, "name": "Truck"}
        ]
    }
    
    current_image_id = 1
    current_ann_id = 1
    
    for dataset, label_file in label_files.items():
        if not os.path.exists(label_file):
            print(f"Warning: Label file {label_file} does not exist, skipping...")
            continue
            
        print(f"Processing {dataset} from {label_file}")
        
        try:
            with open(label_file, 'rb') as f:
                data = orjson.loads(f.read())
            
            # Create mapping from old image IDs to new image IDs
            old_to_new_image_id = {}
            
            # Process images
            for img in data.get("images", []):
                old_id = img["id"]
                new_img = img.copy()
                new_img["id"] = current_image_id
                old_to_new_image_id[old_id] = current_image_id
                merged_coco["images"].append(new_img)
                current_image_id += 1
            
            # Process annotations
            for ann in data.get("annotations", []):
                old_image_id = ann["image_id"]
                if old_image_id in old_to_new_image_id:
                    new_ann = ann.copy()
                    new_ann["id"] = current_ann_id
                    new_ann["image_id"] = old_to_new_image_id[old_image_id]
                    merged_coco["annotations"].append(new_ann)
                    current_ann_id += 1
            
            print(f"Added {len(data.get('images', []))} images and {len(data.get('annotations', []))} annotations from {dataset}")
            
        except Exception as e:
            print(f" Error processing {label_file}: {e}")
            continue
    
    # Save merged file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        f.write(orjson.dumps(merged_coco, option=orjson.OPT_INDENT_2))
    
    print(f"Merged labels saved to {output_file}")
    print(f"Total: {len(merged_coco['images'])} images, {len(merged_coco['annotations'])} annotations")
    
    return output_file

def clean_labels_file(input_file, output_file):
    """Remove images and annotations with '_E_' or '_N_' in filename from a single label file and reassign IDs."""
    print(f"Cleaning labels file: {input_file}")
    
    try:
        with open(input_file, 'rb') as f:
            data = orjson.loads(f.read())
        
        # Filter images and reassign IDs
        original_image_count = len(data.get("images", []))
        filtered_images = []
        old_to_new_image_id = {}
        current_image_id = 1
        
        for img in data.get("images", []):
            filename = img.get("file_name", "")
            
            # Skip images with '_E_' or '_N_' in filename
            if '_E_' in filename or '_N_' in filename:
                continue
            
            # Create new image with reassigned ID
            old_id = img["id"]
            new_img = img.copy()
            new_img["id"] = current_image_id
            old_to_new_image_id[old_id] = current_image_id
            
            filtered_images.append(new_img)
            current_image_id += 1
        
        # Filter annotations and reassign IDs
        original_ann_count = len(data.get("annotations", []))
        filtered_annotations = []
        current_ann_id = 1
        
        for ann in data.get("annotations", []):
            old_image_id = ann["image_id"]
            
            # Only keep annotations for remaining images
            if old_image_id in old_to_new_image_id:
                new_ann = ann.copy()
                new_ann["id"] = current_ann_id
                new_ann["image_id"] = old_to_new_image_id[old_image_id]
                filtered_annotations.append(new_ann)
                current_ann_id += 1
        
        # Create cleaned data
        cleaned_data = {
            "images": filtered_images,
            "annotations": filtered_annotations,
            "categories": data.get("categories", [])
        }
        
        # Save cleaned file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            f.write(orjson.dumps(cleaned_data, option=orjson.OPT_INDENT_2))
        
        removed_image_count = original_image_count - len(filtered_images)
        removed_ann_count = original_ann_count - len(filtered_annotations)
        
        print(f"Cleaned labels saved to {output_file}")
        print(f"Removed {removed_image_count} images and {removed_ann_count} annotations")
        print(f"Reassigned IDs for {len(filtered_images)} images and {len(filtered_annotations)} annotations")
        print(f"Final: {len(filtered_images)} images, {len(filtered_annotations)} annotations")
        
    except Exception as e:
        print(f"Error cleaning {input_file}: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate pseudo labels and nighttime images')
    parser.add_argument('--codetr', type=str, required=True,
                       help='Path to CODETR model checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for pseudo labels and images')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run inference on (e.g., "cuda:0" or "cpu")')
    args = parser.parse_args()
    
    DAY_DIR = os.path.join(args.output_dir, 'day')
    NIGHT_DIR = os.path.join(args.output_dir, 'night')
    
    TRAIN_OUTPUT_DIR = os.path.join(DAY_DIR, 'train')
    TEST_OUTPUT_DIR = os.path.join(DAY_DIR, 'test')
    
    check_data_integrity(TRAIN_DATA, TRAIN_OUTPUT_DIR)
    check_data_integrity(TEST_DATA, TEST_OUTPUT_DIR)
    
    # Generate pseudo labels for training and testing datasets
    TRAIN_LABELS = generate_labels(args, TRAIN_DATA, TRAIN_OUTPUT_DIR)
    TEST_LABELS = generate_labels(args, TEST_DATA, TEST_OUTPUT_DIR)
    
    print("Generated labels for training and testing datasets.")
    print("Training labels:", TRAIN_LABELS)
    print("Testing labels:", TEST_LABELS)
    
    #Merge labels for training and testing datasets
    print("Merging label files...")
    
    # Merge all training labels into one file
    merged_train_labels = os.path.join(TRAIN_OUTPUT_DIR, 'train_before_filter.json')
    merge_all_labels(TRAIN_LABELS, merged_train_labels)
    
    # Merge all test labels into one file
    merged_test_labels = os.path.join(TEST_OUTPUT_DIR, 'test_before_filter.json')
    merge_all_labels(TEST_LABELS, merged_test_labels)
        

    # Copy images to output directories
    print("Copying images to output directories...")
    TRAIN_IMAGES_DIR = os.path.join(TRAIN_OUTPUT_DIR, 'images')
    os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
    for dataset, label_file in TRAIN_LABELS.items():
        with open(label_file, 'rb') as f:
            labels = orjson.loads(f.read())
        images_dir = TRAIN_DATA[dataset]["images_dir"]
        for img in tqdm(labels.get("images", [])):
            img_path = os.path.join(images_dir, img["file_name"])
            target_path = os.path.join(TRAIN_IMAGES_DIR, img["file_name"])
            if os.path.exists(img_path):
                fast_copy(img_path, target_path)
            else:
                print(f"Warning: Image {img_path} does not exist, skipping.")
                
    TEST_IMAGES_DIR = os.path.join(TEST_OUTPUT_DIR, 'images')
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    for dataset, label_file in TEST_LABELS.items():
        with open(label_file, 'rb') as f:
            labels = orjson.loads(f.read())
        images_dir = TEST_DATA[dataset]["images_dir"]
        for img in tqdm(labels.get("images", [])):
            img_path = os.path.join(images_dir, img["file_name"])
            target_path = os.path.join(TEST_IMAGES_DIR, img["file_name"])
            if os.path.exists(img_path):
                fast_copy(img_path, target_path)
            else:
                print(f"Warning: Image {img_path} does not exist, skipping.")
    
    
    # Generate nighttime images
    os.makedirs(NIGHT_DIR, exist_ok=True)
    NIGHT_TRAIN_DIR = os.path.join(NIGHT_DIR, 'train')
    NIGHT_TEST_DIR = os.path.join(NIGHT_DIR, 'test')
    os.makedirs(NIGHT_TRAIN_DIR, exist_ok=True)
    os.makedirs(NIGHT_TEST_DIR, exist_ok=True)
    NIGHT_TRAIN_IMAGES_DIR = os.path.join(NIGHT_TRAIN_DIR, 'images')
    NIGHT_TEST_IMAGES_DIR = os.path.join(NIGHT_TEST_DIR, 'images')
    os.makedirs(NIGHT_TRAIN_IMAGES_DIR, exist_ok=True)
    os.makedirs(NIGHT_TEST_IMAGES_DIR, exist_ok=True)
    
    fast_copy(merged_train_labels, os.path.join(TRAIN_OUTPUT_DIR, 'train.json'))
    fast_copy(merged_test_labels, os.path.join(TRAIN_OUTPUT_DIR, 'test.json'))
    
    print("Starting nighttime image generation for training data...")
    generate_nighttime_images(
        day_images_dir=TRAIN_IMAGES_DIR,
        night_images_dir=NIGHT_TRAIN_IMAGES_DIR,
        device=args.device,
        use_fp16=True,
        labels_dict=TRAIN_LABELS,
    )
    
    # Generate nighttime images for test data
    print("Starting nighttime image generation for test data...")
    generate_nighttime_images(
        day_images_dir=TEST_IMAGES_DIR,
        night_images_dir=NIGHT_TEST_IMAGES_DIR,
        device=args.device,
        use_fp16=True,
        labels_dict=TEST_LABELS,
    )
    
    # Remove fisheye8k nighttime images in the day directory
    clean_labels_file(merged_train_labels, os.path.join(NIGHT_TRAIN_DIR, 'train.json'))
    clean_labels_file(merged_test_labels, os.path.join(NIGHT_TEST_DIR, 'test.json'))
    
    print("Nighttime images generated and labels cleaned.")
    print("All done! Day and night datasets generated.")
    print(f"Day dataset: {DAY_DIR}")
    print(f"Night dataset: {NIGHT_DIR}")
    
if __name__ == '__main__':
    main()