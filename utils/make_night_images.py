import os
import orjson
import torch
import random
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
from tqdm import tqdm
from torchvision import transforms
from .cyclegan_turbo import CycleGAN_Turbo
from .training_utils import build_transform

def is_valid_image(path):
    """Check if image is valid and not corrupted."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img = img.convert('RGB')
        return True
    except:
        return False

def add_flash(img, bbox, flash, alpha=0.5, scale=1.0):
    """Add flash effect to image at bbox location."""
    x, y, w, h = bbox
    size = int(max(w, h) * scale)
    flash_resized = flash.resize((size, size), Image.LANCZOS)

    y_indices, x_indices = np.ogrid[:size, :size]
    center = (size - 1) / 2
    distance = np.sqrt((x_indices - center) ** 2 + (y_indices - center) ** 2)
    max_distance = center
    
    mask_array = (1 - (distance / max_distance)) * 255 * alpha
    mask_array = np.clip(mask_array, 0, 255).astype(np.uint8)
    mask = Image.fromarray(mask_array, mode='L')

    flash_circle = flash_resized.copy()
    flash_circle.putalpha(mask)

    center_x = int(x + w / 2)
    center_y = int(y + h / 2)
    paste_x = center_x - size // 2
    paste_y = center_y - size // 2

    img = img.convert('RGBA')
    img.paste(flash_circle, (paste_x, paste_y), flash_circle)
    return img.convert('RGB')

def add_motion_blur(img, bbox, kernel_size=10, direction='horizontal'):
    """Add motion blur to image at bbox location."""
    x, y, w, h = map(int, bbox)
    img_w, img_h = img.size

    # Clamp bbox to image bounds
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))

    if w <= 1 or h <= 1:
        return img

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    roi = img_cv[y:y+h, x:x+w]

    if roi.size == 0:
        return img

    kernel = np.zeros((kernel_size, kernel_size))
    if direction == 'vertical':
        kernel[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    else:
        kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel /= kernel_size

    blurred_roi = cv2.filter2D(roi, -1, kernel)
    img_cv[y:y+h, x:x+w] = blurred_roi
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return img_pil

def edit_brightness(img, brightness_factor=None, grayscale=True):
    """Edit image brightness and convert to grayscale."""
    if grayscale:
        img = img.convert('L')
    if brightness_factor is not None:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
    return img

def apply_light_augmentation(img, bboxes, flash=None):
    """Apply light augmentation to a single image."""
    # Apply brightness adjustment and grayscale
    img = edit_brightness(img, brightness_factor=random.uniform(1, 2), grayscale=True)
    
    # Apply flash and motion blur effects to bboxes
    for bbox in bboxes:
        x, y, w, h = map(int, bbox)
        
        # Add flash effect (20% chance for objects > 1500 pixels)
        if flash and random.random() <= 0.2:
            if w * h > 1500:
                img = add_flash(img, bbox, flash, alpha=random.uniform(0.5, 1.4), scale=0.7)
        
        # Add motion blur (75% chance for objects > 3000 pixels)
        if random.random() <= 0.75:
            if w * h > 3000:
                img = add_motion_blur(img, bbox, kernel_size=10, direction='horizontal')
    
    return img

class BBoxCache:
    """Cache for bounding boxes to avoid repeated file reads."""
    def __init__(self, labels_dict):
        self.bbox_cache = {}
        self._load_all_bboxes(labels_dict)
    
    def _load_all_bboxes(self, labels_dict):
        """Load all bounding boxes into memory once."""
        if not labels_dict:
            return
            
        for dataset_labels in labels_dict.values():
            if os.path.exists(dataset_labels):
                try:
                    with open(dataset_labels, 'rb') as f:
                        data = orjson.loads(f.read())
                    
                    # Create image filename to ID mapping
                    img_name_to_id = {img['file_name']: img['id'] for img in data.get('images', [])}
                    
                    # Group annotations by image ID
                    annotations_by_image = {}
                    for ann in data.get('annotations', []):
                        img_id = ann['image_id']
                        if img_id not in annotations_by_image:
                            annotations_by_image[img_id] = []
                        annotations_by_image[img_id].append(ann)
                    
                    # Cache bboxes by filename
                    for img_name, img_id in img_name_to_id.items():
                        if img_id in annotations_by_image:
                            self.bbox_cache[img_name] = [
                                ann['bbox'] for ann in annotations_by_image[img_id]
                                if ann['category_id'] != 3  # non-pedestrian objects
                            ]
                        else:
                            self.bbox_cache[img_name] = []
                            
                except Exception as e:
                    print(f"Error loading labels from {dataset_labels}: {e}")
                    continue
    
    def get_bboxes(self, img_name):
        """Get cached bounding boxes for an image."""
        return self.bbox_cache.get(img_name, [])

def generate_nighttime_images(day_images_dir, night_images_dir, device="cpu", use_fp16=True,
                            labels_dict=None, flash_path="utils/flash.png", apply_augmentation=True, 
                            batch_size=32):
    """Generate nighttime images from day images with batch processing."""
    print(f"Generating nighttime images from {day_images_dir} to {night_images_dir}")
    
    # Load model
    print("Loading day-to-night model...")
    model = CycleGAN_Turbo(pretrained_name="day_to_night")
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()
    if use_fp16:
        model.half()
    model.to(device)
    T_val = build_transform("resize_512x512")
    print("Model loaded.")
    
    # Load flash image if provided
    flash = None
    if flash_path and os.path.exists(flash_path):
        flash = Image.open(flash_path).convert('RGBA')
        print(f"Flash image loaded from {flash_path}")
    elif apply_augmentation:
        print("No flash image provided, skipping flash effects")
    
    # Create bbox cache for fast lookup
    print("Loading bounding box cache...")
    bbox_cache = BBoxCache(labels_dict) if labels_dict else None
    print("Bounding box cache loaded.")
    
    # Create output directory
    os.makedirs(night_images_dir, exist_ok=True)
    
    # Get all image files efficiently
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = [f for f in Path(day_images_dir).iterdir() 
                   if f.suffix.lower() in valid_extensions]
    
    print(f"Found {len(image_files)} images to convert.")
    
    # Process images one by one to avoid batch shape issues
    processed = 0
    for img_file in tqdm(image_files, desc="🌙 Converting"):
        output_path = Path(night_images_dir) / img_file.name
        
        # Skip if exists and valid
        if output_path.exists() and is_valid_image(output_path):
            continue
            
        # Remove corrupted file if exists
        if output_path.exists() and not is_valid_image(output_path):
            try:
                output_path.unlink()
            except:
                continue
        
        try:
            # Load and process single image
            img = Image.open(img_file).convert('RGB')
            original_size = img.size
            
            # Transform
            img_transformed = T_val(img)
            x_t = transforms.ToTensor()(img_transformed)
            x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0)  # Add batch dimension
            
            if use_fp16:
                x_t = x_t.half()
            
            x_t = x_t.to(device)
            
            # Single image inference
            with torch.no_grad():
                output = model(x_t, direction=None, caption=None)
            
            # Process output
            output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
            output_pil = output_pil.resize(original_size, Image.LANCZOS)
            
            # Apply augmentation
            if apply_augmentation and bbox_cache:
                bboxes = bbox_cache.get_bboxes(img_file.name)
                if bboxes:
                    output_pil = apply_light_augmentation(output_pil, bboxes, flash)
            
            output_pil.save(output_path)
            processed += 1
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    print(f"Nighttime images generated in {night_images_dir}")
    print(f"Total processed: {processed} images")
    return night_images_dir