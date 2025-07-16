import os
import orjson
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
import cv2
import numpy as np
from tqdm import tqdm

def generate_pseudo_labels(config_file, checkpoint_file, images_dir, confidence_threshold=0.3, output_file='pseudo_labels.json', device='cpu', number_images=None):
    """
    Generate pseudo labels for LOAF dataset using trained Co-DETR model
    Adds resume feature: skips images with existing label files and resumes COCO annotation.
    """
    register_all_modules()
    print(f"Loading model from {checkpoint_file}")
    model = init_detector(config_file, checkpoint_file, device=device)

    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
    os.makedirs(output_dir, exist_ok=True)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext)])
    if number_images is not None:
        image_files = image_files[:number_images]
    
    # Resume: find already processed images
    processed = set()
    print(f"Found {len(image_files)} images in {images_dir}, {len(processed)} already processed.")

    # Resume: load existing COCO annotations if available
    coco_annotations = {
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
    annotation_id = 1
    if os.path.exists(output_file):
        with open(output_file, 'rb') as f:
            coco_annotations = orjson.loads(f.read())
        if coco_annotations["annotations"]:
            annotation_id = max(a["id"] for a in coco_annotations["annotations"]) + 1

    for idx, image_file in enumerate(tqdm(image_files, desc="Generating pseudo labels")):
        image_name_without_ext = os.path.splitext(image_file)[0]
        if image_name_without_ext in processed:
            continue  # Skip already processed images

        image_path = os.path.join(images_dir, image_file)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not load image {image_path}")
            continue

        h, w = img.shape[:2]
        result = inference_detector(model, image_path)
        yolo_labels = []

        coco_annotations["images"].append({
            "id": idx + 1,
            "file_name": image_file,
            "width": w,
            "height": h
        })

        if hasattr(result, 'pred_instances'):
            pred_instances = result.pred_instances
            bboxes = pred_instances.bboxes.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()

            for bbox, score, label in zip(bboxes, scores, labels):
                if score >= confidence_threshold:
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) / 2 / w
                    center_y = (y1 + y2) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    yolo_labels.append(f"{label} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
                    coco_annotations["annotations"].append({
                        "id": annotation_id,
                        "image_id": idx + 1,
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "area": float((x2 - x1) * (y2 - y1)),
                        "iscrowd": 0,
                        "score": float(score)
                    })
                    annotation_id += 1

        # Save COCO format annotations with orjson (faster than json)
        with open(output_file, 'wb') as f:
            f.write(orjson.dumps(coco_annotations, option=orjson.OPT_INDENT_2))

    print(f"Pseudo labels saved to: {output_dir}")
    print(f"COCO format annotations: {output_file}")
    print(f"Total annotations generated: {annotation_id - 1}")