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

def get_bboxes_for_image(img_name, labels_dict):
    """Get bounding boxes for a specific image from labels."""
    bboxes = []
    for dataset_labels in labels_dict.values():
        if os.path.exists(dataset_labels):
            try:
                with open(dataset_labels, 'rb') as f:
                    data = orjson.loads(f.read())
                
                # Find image ID
                img_id = None
                for image_info in data.get('images', []):
                    if image_info['file_name'] == img_name:
                        img_id = image_info['id']
                        break
                
                if img_id is not None:
                    # Get bboxes for non-pedestrian objects
                    bboxes = [
                        ann['bbox'] for ann in data.get('annotations', [])
                        if ann['image_id'] == img_id and ann['category_id'] != 3
                    ]
                    break
            except Exception as e:
                print(f"Error reading labels from {dataset_labels}: {e}")
                continue
    
    return bboxes

def convert_to_night(input_path, output_path, model, T_val, device, use_fp16=False, 
                    labels_dict=None, flash=None, apply_augmentation=True):
    """Convert a single image from day to night with optional augmentation."""
    # Skip if night image already exists and is valid
    if output_path.exists() and is_valid_image(output_path):
        return
    
    # Remove corrupted file if exists
    if output_path.exists() and not is_valid_image(output_path):
        try:
            output_path.unlink()
        except Exception as e:
            print(f"Error removing corrupted file {output_path}: {e}")
            return
    
    try:
        input_image = Image.open(input_path).convert('RGB')
        input_img = T_val(input_image)
        x_t = transforms.ToTensor()(input_img)
        x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).to(device)
        
        if use_fp16:
            x_t = x_t.half()
        
        with torch.no_grad():
            output = model(x_t, direction=None, caption=None)
        
        output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)
        
        # Apply augmentation before saving
        if apply_augmentation and labels_dict:
            img_name = input_path.name
            bboxes = get_bboxes_for_image(img_name, labels_dict)
            if bboxes:
                output_pil = apply_light_augmentation(output_pil, bboxes, flash)
        
        output_pil.save(output_path)
        
    except Exception as e:
        print(f"Error converting {input_path} to night: {e}")

def generate_nighttime_images(day_images_dir, night_images_dir, device="cpu", use_fp16=False,
                            labels_dict=None, flash_path='utils/flash.png', apply_augmentation=True):
    """Generate nighttime images from day images with optional augmentation."""
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
        
    
    # Create output directory
    os.makedirs(night_images_dir, exist_ok=True)
    
    # Get all image files
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    image_files = []
    for ext in valid_extensions:
        image_files.extend(Path(day_images_dir).glob(f"*{ext}"))
        image_files.extend(Path(day_images_dir).glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images to convert.")
    
    # Convert each image
    for img_path in tqdm(image_files, desc="Converting to night"):
        output_path = Path(night_images_dir) / img_path.name
        convert_to_night(img_path, output_path, model, T_val, device, use_fp16,
                        labels_dict, flash, apply_augmentation)
    
    print(f"Nighttime images generated in {night_images_dir}")
    return night_images_dir