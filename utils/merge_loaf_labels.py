import os
import numpy as np
import orjson
from tqdm import tqdm

def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def merge_loaf_labels(pseudo_file, real_file, output_file, iou_threshold=0.2):
    """
    Merge pseudo LOAF labels with ground truth labels
    
    Args:
        pseudo_file: Path to pseudo labels JSON file
        real_file: Path to ground truth labels JSON file  
        output_file: Path to output merged labels JSON file
        iou_threshold: IoU threshold for overlap detection
    """
    print(f"Loading pseudo labels from {pseudo_file}")
    with open(pseudo_file, "rb") as f:
        pseudo = orjson.loads(f.read())
    
    print(f"Loading real labels from {real_file}")  
    with open(real_file, "rb") as f:
        real = orjson.loads(f.read())

    # Create filename to image mappings
    pseudo_imgs = {img["file_name"]: img for img in pseudo.get("images", [])}
    real_imgs = {img["file_name"]: img for img in real.get("images", [])}

    pseudo_fn2id = {img["file_name"]: img["id"] for img in pseudo.get("images", [])}
    real_fn2id = {img["file_name"]: img["id"] for img in real.get("images", [])}
    real_id2fn = {img["id"]: img["file_name"] for img in real.get("images", [])}

    # Group pseudo annotations by filename and category
    pseudo_bikes = {}
    pseudo_pedestrian = {}
    for ann in pseudo["annotations"]:
        fn = next((fn for fn, iid in pseudo_fn2id.items() if iid == ann["image_id"]), None)
        if fn is None: 
            continue
        if ann["category_id"] == 1:  # Bike
            pseudo_bikes.setdefault(fn, []).append(ann["bbox"])
        elif ann["category_id"] == 3:  # Pedestrian
            pseudo_pedestrian.setdefault(fn, []).append(ann["bbox"])

    merged_anns = []
    ann_id = 1

    # Keep all pseudo annotations
    print("Adding pseudo annotations...")
    for ann in pseudo["annotations"]:
        merged_anns.append({**ann, "id": ann_id})
        ann_id += 1

    # Add real annotations with overlap filtering
    print("Adding real annotations with overlap filtering...")
    for ann in tqdm(real["annotations"]):
        real_fn = real_id2fn.get(ann["image_id"])
        if real_fn is None or real_fn not in pseudo_fn2id:
            continue

        pseudo_img_id = pseudo_fn2id[real_fn]
        cat_id = ann["category_id"]

        # For non-person categories, add directly
        if cat_id != 1:  # Not person category
            merged_anns.append({
                **ann, 
                "image_id": pseudo_img_id, 
                "id": ann_id
            })
            ann_id += 1
            continue

        # For person category, check overlap with bikes and pedestrians
        overlap = any(
            bbox_iou(ann["bbox"], bb) > iou_threshold
            for bb in pseudo_bikes.get(real_fn, []) + pseudo_pedestrian.get(real_fn, [])
        )
        
        if not overlap:
            # Convert person to pedestrian category and add
            merged_anns.append({
                **ann,
                "image_id": pseudo_img_id,
                "category_id": 3,  # Convert to pedestrian
                "id": ann_id
            })
            ann_id += 1

    # Create merged dataset
    merged = {
        "images": list(pseudo_imgs.values()),
        "annotations": merged_anns,
        "categories": pseudo.get("categories", real.get("categories", []))
    }

    # Save merged dataset with orjson (handles numpy types natively)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        f.write(orjson.dumps(merged, option=orjson.OPT_INDENT_2))
    
    print(f"Saved merged labels to {output_file}")
    print(f"Total merged annotations: {len(merged_anns)}")
    return output_file