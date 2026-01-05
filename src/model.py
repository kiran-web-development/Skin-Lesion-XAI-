# model.py: ResNet-18 for ISIC 2019

import torch
import torch.nn as nn
import torchvision.models as models

# Try to import from current directory first, then fallback
try:
    from config import NUM_CLASSES
except ImportError:
    from .config import NUM_CLASSES

def get_resnet18_model(pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return model

# For transfer learning: freeze all except final layer

def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

# For fine-tuning: unfreeze all

def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True
