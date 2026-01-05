# explain_lime.py: LIME explanations for ISIC 2019 predictions

import os
import torch
import numpy as np
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from config import models_dir, lime_dir, device, CLASSES, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
from model import get_resnet18_model
from torchvision import transforms
from utils import ensure_dir

def get_preprocess():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def predict_fn(images):
    # images: numpy array (N, H, W, C), range [0,255]
    model.eval()
    batch = []
    for img in images:
        img = Image.fromarray(img.astype(np.uint8))
        tensor = preprocess(img).unsqueeze(0).to(device)
        batch.append(tensor)
    batch = torch.cat(batch, dim=0)
    with torch.no_grad():
        outputs = model(batch)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    return probs

if __name__ == '__main__':
    ensure_dir(lime_dir)
    # Load model
    model = get_resnet18_model(pretrained=False)
    model.load_state_dict(torch.load(os.path.join(models_dir, 'best_model.pt'), map_location=device))
    model = model.to(device)
    preprocess = get_preprocess()

    # Example: explain a few images from each class
    for cls in CLASSES:
        cls_dir = os.path.join('..', 'data', 'raw', 'ISIC2019', cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir)[:2]:  # 2 samples per class
            img_path = os.path.join(cls_dir, fname)
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                img_np,
                classifier_fn=predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=1000
            )
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0], positive_only=True, num_features=8, hide_rest=False
            )
            plt.figure(figsize=(5,5))
            plt.imshow(mark_boundaries(temp, mask))
            plt.title(f'LIME: {cls} - {fname}')
            plt.axis('off')
            out_path = os.path.join(lime_dir, f'{cls}_{fname}_lime.png')
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()
