

import sys
import os
import torch
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, CLASSES
from torchvision import transforms

def test_image_processing(image_path):
    """Test image processing pipeline"""
    
    print(f"📸 Testing image: {image_path}")
    print("=" * 70)
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f" ERROR: File not found: {image_path}")
        return
    
    file_size = os.path.getsize(image_path) / 1024
    print(f" File exists - Size: {file_size:.2f} KB")
    
    try:
        # Try to open image
        img = Image.open(image_path)
        print(f" Image opened successfully")
        print(f"  - Format: {img.format}")
        print(f"  - Mode: {img.mode}")
        print(f"  - Size: {img.size}")
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            print(f" Converting from {img.mode} to RGB...")
            img = img.convert('RGB')
        
        # Check image dimensions
        if img.size[0] < 100 or img.size[1] < 100:
            print(f" WARNING: Image is very small ({img.size[0]}x{img.size[1]})")
        
        # Test preprocessing pipeline
        preprocess = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        print(f"\n Preprocessing Pipeline:")
        print(f"  - Resize to: {IMG_SIZE}x{IMG_SIZE}")
        print(f"  - ToTensor: Yes")
        print(f"  - Normalize: Yes (ImageNet stats)")
        
        tensor = preprocess(img)
        print(f" Preprocessing successful")
        print(f"  - Output shape: {tensor.shape}")
        print(f"  - Data type: {tensor.dtype}")
        print(f"  - Min value: {tensor.min():.4f}")
        print(f"  - Max value: {tensor.max():.4f}")
        
        # Test with batch
        batch = tensor.unsqueeze(0)
        print(f"\n Batch creation successful")
        print(f"  - Batch shape: {batch.shape}")
        
        print(f"\n Image processing test PASSED!")
        print(f"   The image can be processed by the model successfully.")
        
    except Exception as e:
        print(f"\n ERROR during image processing:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    image_path = r"C:\Users\saini\Downloads\original.jpg"
    test_image_processing(image_path)
