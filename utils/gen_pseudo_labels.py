import os
import orjson
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

def generate_pseudo_labels(config_file, checkpoint_file, images_dir, confidence_threshold=0.3, 
                         output_file='pseudo_labels.json', device='cpu', number_images=None, 
                         batch_size=32, save_interval=100):
    """
    Generate pseudo labels for LOAF dataset using trained Co-DETR model
    Optimized version with batching and reduced I/O operations.
    """
    register_all_modules()
    print(f"Loading model from {checkpoint_file}")
    model = init_detector(config_file, checkpoint_file, device=device)

    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files at once
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext)])
    
    if number_images is not None:
        image_files = image_files[:number_images]
    
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
    processed_images = set()
    
    if os.path.exists(output_file):
        with open(output_file, 'rb') as f:
            coco_annotations = orjson.loads(f.read())
        if coco_annotations["annotations"]:
            annotation_id = max(a["id"] for a in coco_annotations["annotations"]) + 1
        # Get already processed images
        processed_images = {img["file_name"] for img in coco_annotations["images"]}

    print(f"Found {len(image_files)} images in {images_dir}, {len(processed_images)} already processed.")
    
    # Filter out already processed images
    remaining_images = [img for img in image_files if img not in processed_images]
    print(f"Processing {len(remaining_images)} remaining images...")

    # Process images in batches
    for batch_start in tqdm(range(0, len(remaining_images), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(remaining_images))
        batch_images = remaining_images[batch_start:batch_end]
        
        # Process batch
        batch_coco_images = []
        batch_coco_annotations = []
        
        for i, image_file in enumerate(batch_images):
            image_path = os.path.join(images_dir, image_file)
            
            # Use cv2.imread for faster loading
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not load image {image_path}")
                continue

            h, w = img.shape[:2]
            
            # Get current image ID
            current_image_id = len(coco_annotations["images"]) + len(batch_coco_images) + 1
            
            batch_coco_images.append({
                "id": current_image_id,
                "file_name": image_file,
                "width": w,
                "height": h
            })
            
            # Run inference
            result = inference_detector(model, image_path)
            
            if hasattr(result, 'pred_instances'):
                pred_instances = result.pred_instances
                bboxes = pred_instances.bboxes.cpu().numpy()
                scores = pred_instances.scores.cpu().numpy()
                labels = pred_instances.labels.cpu().numpy()
                
                # Vectorized filtering
                valid_indices = scores >= confidence_threshold
                
                if np.any(valid_indices):
                    valid_bboxes = bboxes[valid_indices]
                    valid_scores = scores[valid_indices]
                    valid_labels = labels[valid_indices]
                    
                    for bbox, score, label in zip(valid_bboxes, valid_scores, valid_labels):
                        x1, y1, x2, y2 = bbox
                        batch_coco_annotations.append({
                            "id": annotation_id,
                            "image_id": current_image_id,
                            "category_id": int(label),
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "area": float((x2 - x1) * (y2 - y1)),
                            "iscrowd": 0,
                            "score": float(score)
                        })
                        annotation_id += 1
        
        # Add batch results to main annotations
        coco_annotations["images"].extend(batch_coco_images)
        coco_annotations["annotations"].extend(batch_coco_annotations)
        
        # Save periodically to avoid losing progress
        if (batch_start // batch_size) % save_interval == 0:
            with open(output_file, 'wb') as f:
                f.write(orjson.dumps(coco_annotations, option=orjson.OPT_INDENT_2))
            print(f"Saved progress: {len(coco_annotations['images'])} images processed")

    # Final save
    with open(output_file, 'wb') as f:
        f.write(orjson.dumps(coco_annotations, option=orjson.OPT_INDENT_2))

    print(f"Pseudo labels saved to: {output_file}")
    print(f"Total images processed: {len(coco_annotations['images'])}")
    print(f"Total annotations generated: {len(coco_annotations['annotations'])}")

def process_images_parallel(image_batch, model, images_dir, confidence_threshold, start_image_id, start_ann_id):
    """Process a batch of images in parallel."""
    batch_coco_images = []
    batch_coco_annotations = []
    current_image_id = start_image_id
    current_ann_id = start_ann_id
    
    for image_file in image_batch:
        image_path = os.path.join(images_dir, image_file)
        
        img = cv2.imread(image_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        
        batch_coco_images.append({
            "id": current_image_id,
            "file_name": image_file,
            "width": w,
            "height": h
        })
        
        result = inference_detector(model, image_path)
        
        if hasattr(result, 'pred_instances'):
            pred_instances = result.pred_instances
            bboxes = pred_instances.bboxes.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()
            
            valid_indices = scores >= confidence_threshold
            
            if np.any(valid_indices):
                valid_bboxes = bboxes[valid_indices]
                valid_scores = scores[valid_indices]
                valid_labels = labels[valid_indices]
                
                for bbox, score, label in zip(valid_bboxes, valid_scores, valid_labels):
                    x1, y1, x2, y2 = bbox
                    batch_coco_annotations.append({
                        "id": current_ann_id,
                        "image_id": current_image_id,
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "area": float((x2 - x1) * (y2 - y1)),
                        "iscrowd": 0,
                        "score": float(score)
                    })
                    current_ann_id += 1
        
        current_image_id += 1
    
    return batch_coco_images, batch_coco_annotations