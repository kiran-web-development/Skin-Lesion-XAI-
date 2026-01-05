# preprocessing.py: Preprocessing pipeline for ISIC 2019 images

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from config import raw_data_dir, processed_data_dir, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
from utils import ensure_dir


import torch
import torchvision.transforms as T

def crop_roi(image):
    # Automatic ROI cropping: find bounding box of non-background
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = image[y:y+h, x:x+w]
    return cropped

def pad_to_square(image):
    h, w, c = image.shape
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    padded = cv2.copyMakeBorder(image, pad_h, size-h-pad_h, pad_w, size-w-pad_w, cv2.BORDER_CONSTANT, value=[0,0,0])
    return padded

def preprocess_image(img_path, save_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_roi(image)
    image = pad_to_square(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = Image.fromarray(image)
    # Normalize using ImageNet stats
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    image = transform(image)
    # Save as .pt tensor
    torch.save(image, save_path)

def preprocess_all():
    ensure_dir(processed_data_dir)
    for cls in os.listdir(raw_data_dir):
        cls_dir = os.path.join(raw_data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        save_cls_dir = os.path.join(processed_data_dir, cls)
        ensure_dir(save_cls_dir)
        for fname in tqdm(os.listdir(cls_dir), desc=f'Preprocessing {cls}'):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(cls_dir, fname)
            save_path = os.path.join(save_cls_dir, fname.replace('.jpg', '.pt').replace('.jpeg', '.pt').replace('.png', '.pt'))
            preprocess_image(img_path, save_path)

if __name__ == '__main__':
    preprocess_all()
