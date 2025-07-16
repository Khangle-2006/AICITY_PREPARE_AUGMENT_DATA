import orjson
import os
from collections import defaultdict

def filter_loaf(input_file, output_file, top_k=20000):
    """
    Filter top K images from LOAF dataset based on non-pedestrian objects,
    prioritizing truck and bike detections.
    """
    with open(input_file, "rb") as f:
        coco = orjson.loads(f.read())
    
    counts = defaultdict(lambda: {"non_pedestrian": 0, "truck": 0, "bus": 0, "car": 0, "bike": 0})
    
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        
        if cat_id != 3:
            counts[img_id]["non_pedestrian"] += 1
        if cat_id == 0:
            counts[img_id]["bus"] += 1
        if cat_id == 1:
            counts[img_id]["bike"] += 1
        if cat_id == 2:
            counts[img_id]["car"] += 1
        if cat_id == 4:
            counts[img_id]["truck"] += 1
    
    def sorting_key(img):
        c = counts[img["id"]]
        return (c["non_pedestrian"], c["truck"], c["bus"], c["car"], c["bike"])
    
    sorted_images = sorted(coco["images"], key=sorting_key, reverse=True)
    
    selected_ids = set()
    selected_images = []
    
    for img in sorted_images:
        c = counts[img["id"]]
        if c["truck"] > 0 or c["bike"] > 0:
            selected_images.append(img)
            selected_ids.add(img["id"])
    
    for img in sorted_images:
        if len(selected_images) >= top_k:
            break
        if img["id"] not in selected_ids:
            c = counts[img["id"]]
            if c["non_pedestrian"] > 0:
                selected_images.append(img)
                selected_ids.add(img["id"])
    
    for img in sorted_images:
        if len(selected_images) >= top_k:
            break
        if img["id"] not in selected_ids:
            selected_images.append(img)
            selected_ids.add(img["id"])
    
    print(f"Total selected images: {len(selected_images)}")
    
    new_coco = coco.copy()
    new_coco["images"] = selected_images
    new_coco["annotations"] = [ann for ann in coco["annotations"] if ann["image_id"] in selected_ids]
    
    with open(output_file, "wb") as f:
        f.write(orjson.dumps(new_coco, option=orjson.OPT_INDENT_2))
    
    print(f"Filtered COCO dataset saved to {output_file}")
    return output_file