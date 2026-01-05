# evaluate.py: Evaluation script for ISIC 2019 ResNet-18

import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import models_dir, device, CLASSES, BATCH_SIZE
from dataset import ISIC2019Dataset
from model import get_resnet18_model
from augmentation import val_transforms

# Load validation/test set
val_dataset = ISIC2019Dataset(split='val', transform=None)
# On Windows set num_workers=0 to avoid multiprocessing permission errors
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Load model
model = get_resnet18_model(pretrained=False)
model.load_state_dict(torch.load(os.path.join(models_dir, 'best_model.pt'), map_location=device))
model = model.to(device)
model.eval()

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    total = len(val_loader)
    for i, (images, labels) in enumerate(val_loader):
        if i % 20 == 0:
            print(f"[EVAL] batch {i}/{total}")
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Metrics
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='macro')
rec = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
cm = confusion_matrix(all_labels, all_preds)

print(f"Accuracy: {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall: {rec*100:.2f}%")
print(f"F1-Score: {f1*100:.2f}%")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASSES))

# Confusion matrix plot
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(models_dir, 'confusion_matrix.png'))
plt.close()
