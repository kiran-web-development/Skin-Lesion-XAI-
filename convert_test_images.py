"""
Convert preprocessed .pt tensor files to PNG images for testing
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import processed_data_dir, CLASSES, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

def tensor_to_image(tensor):
    """Convert normalized tensor to PIL Image"""
    # Denormalize
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    # Denormalize
    img_array = tensor.copy()
    img_array = img_array * std + mean
    img_array = np.clip(img_array, 0, 1)
    
    # Convert to uint8
    img_array = (img_array * 255).astype(np.uint8)
    img_array = np.transpose(img_array, (1, 2, 0))
    
    return Image.fromarray(img_array)

def convert_pt_to_png():
    """Convert all .pt files to PNG images in test folder"""
    test_dir = os.path.join(os.path.dirname(__file__), 'test_images')
    os.makedirs(test_dir, exist_ok=True)
    
    count = 0
    for cls in CLASSES:
        cls_dir = os.path.join(processed_data_dir, cls)
        test_cls_dir = os.path.join(test_dir, cls)
        os.makedirs(test_cls_dir, exist_ok=True)
        
        if not os.path.isdir(cls_dir):
            print(f"Skipping {cls} - directory not found")
            continue
        
        # Get first 5 images from each class
        files = sorted(os.listdir(cls_dir))[:5]
        
        for fname in files:
            if not fname.endswith('.pt'):
                continue
            
            pt_path = os.path.join(cls_dir, fname)
            png_fname = fname.replace('.pt', '.png')
            png_path = os.path.join(test_cls_dir, png_fname)
            
            try:
                # Load tensor
                tensor = torch.load(pt_path, map_location='cpu')
                
                # Convert to image
                img = tensor_to_image(tensor)
                
                # Save as PNG
                img.save(png_path)
                print(f"✓ Converted: {cls}/{png_fname}")
                count += 1
            except Exception as e:
                print(f"✗ Error converting {fname}: {e}")
    
    print(f"\n✓ Successfully converted {count} test images to test_images/")
    print(f"  Test images saved to: {test_dir}")
    print(f"  You can now upload images from there to the Flask app!")

if __name__ == '__main__':
    print("Converting .pt tensor files to PNG images...")
    print("=" * 60)
    convert_pt_to_png()
