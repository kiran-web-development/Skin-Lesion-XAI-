# train.py: Training script for ISIC 2019 ResNet-18

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE, device, models_dir, CLASSES
from dataset import ISIC2019Dataset
from model import get_resnet18_model, set_parameter_requires_grad, unfreeze_all_layers
from augmentation import train_transforms, val_transforms
from utils import set_seed, ensure_dir
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight


def main():
    set_seed(42)

    # Prepare datasets
    train_dataset = ISIC2019Dataset(split='train', transform=None, balance=True)
    val_dataset = ISIC2019Dataset(split='val', transform=None)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = get_resnet18_model(pretrained=True)
    set_parameter_requires_grad(model, feature_extracting=False)  # Fine-tune all layers
    model = model.to(device)

    # Compute class weights for imbalance
    labels = np.array(train_dataset.labels)
    class_weights = compute_class_weight('balanced', classes=np.arange(len(CLASSES)), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Loss, optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Early stopping
    best_val_loss = float('inf')
    patience = 0
    train_losses, val_losses, lrs = [], [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        batch_idx = 0
        for images, labels in train_loader:
            if batch_idx % 50 == 0:
                print(f"[TRAIN] Epoch {epoch+1} loading batch {batch_idx}")
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            batch_idx += 1
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        lrs.append(optimizer.param_groups[0]['lr'])

        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            ensure_dir(models_dir)
            torch.save(model.state_dict(), os.path.join(models_dir, 'best_model.pt'))
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    # Plot learning rate vs loss
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Learning Curve')
    plt.savefig(os.path.join(models_dir, 'learning_curve.png'))
    plt.close()


if __name__ == '__main__':
    main()
