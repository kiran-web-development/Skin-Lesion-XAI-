# Configuration for Skin Lesion XAI Project

import os

# Paths
data_dir = os.path.join(os.path.dirname(__file__), '../data')
raw_data_dir = os.path.join(data_dir, 'raw/ISIC2019')
processed_data_dir = os.path.join(data_dir, 'processed')
models_dir = os.path.join(os.path.dirname(__file__), '../models')
lime_dir = os.path.join(os.path.dirname(__file__), '../lime_outputs')

# Classes (as per ISIC 2019)
CLASSES = [
    'AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC'
]
NUM_CLASSES = len(CLASSES)

# Training Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 7

# Image Parameters
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Device
device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') or os.environ.get('CUDA_PATH') else 'cpu'
