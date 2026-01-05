# augmentation.py: Data augmentation pipeline for ISIC 2019

import torchvision.transforms as T
from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

# Data augmentation for training
train_transforms = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(30),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# For validation/test: only resize and normalize
val_transforms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])
