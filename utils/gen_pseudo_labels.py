import os
import orjson
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
import cv2
import numpy as np
from tqdm import tqdm


def generate_pseudo_labels(config_file, checkpoint_file, images_dir, confidence_threshold=0.5, 
                              output_file='pseudo_labels.json', device='cpu', number_images=None, 
                              save_interval=100):

    register_all_modules()
    print(f"Loading model from {checkpoint_file}")
    model = init_detector(config_file, checkpoint_file, device=device)

    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext)])
    
    if number_images is not None:
        image_files = image_files[:number_images]

    # Initialize data structures
    all_images = []
    all_annotations = []
    
    categories = [
        {"id": 0, "name": "Bus"},
        {"id": 1, "name": "Bike"},
        {"id": 2, "name": "Car"},
        {"id": 3, "name": "Pedestrian"},
        {"id": 4, "name": "Truck"}
    ]
    
    annotation_id = 1
    
    print(f"Processing {len(image_files)} images...")
    
    def save_annotations():
        """Helper function to save annotations to file"""
        final_annotations = {
            "images": all_images,
            "annotations": all_annotations,
            "categories": categories
        }
        
        with open(output_file, 'wb') as f:
            f.write(orjson.dumps(final_annotations, option=orjson.OPT_INDENT_2))
    
    # Process all images with periodic saving
    for idx, image_file in enumerate(tqdm(image_files, desc="Generating pseudo labels")):
        image_path = os.path.join(images_dir, image_file)
        
        img = cv2.imread(image_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        current_image_id = idx + 1
        
        all_images.append({
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
            
            # Vectorized filtering
            valid_mask = scores >= confidence_threshold
            
            if np.any(valid_mask):
                valid_bboxes = bboxes[valid_mask]
                valid_scores = scores[valid_mask]
                valid_labels = labels[valid_mask]
                
                # Vectorized annotation creation
                for bbox, score, label in zip(valid_bboxes, valid_scores, valid_labels):
                    x1, y1, x2, y2 = bbox
                    all_annotations.append({
                        "id": annotation_id,
                        "image_id": current_image_id,
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "area": float((x2 - x1) * (y2 - y1)),
                        "iscrowd": 0,
                        "score": float(score)
                    })
                    annotation_id += 1
        
        # Save every save_interval images
        if (idx + 1) % save_interval == 0:
            save_annotations()
            print(f"Saved checkpoint after processing {idx + 1} images")
    
    # Final save
    save_annotations()

    print(f"Pseudo labels saved to: {output_file}")
    print(f"Total images processed: {len(all_images)}")
    print(f"Total annotations generated: {len(all_annotations)}")