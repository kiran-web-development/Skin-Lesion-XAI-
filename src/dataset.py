# dataset.py: ISIC 2019 Dataset Loader

import os
import torch
from torch.utils.data import Dataset
from config import processed_data_dir, CLASSES
from PIL import Image

class ISIC2019Dataset(Dataset):
    def __init__(self, split='train', transform=None, balance=False):
        self.transform = transform
        self.samples = []
        self.labels = []
        self.class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
        split_file = os.path.join(processed_data_dir, f'{split}_split.csv')
        with open(split_file, 'r') as f:
            for line in f:
                img_path, label = line.strip().split(',')
                self.samples.append(img_path)
                self.labels.append(self.class_to_idx[label])
        # Optionally handle class imbalance (oversampling)
        if balance:
            self._balance_dataset()

    def _balance_dataset(self):
        from collections import Counter
        from sklearn.utils import resample
        counter = Counter(self.labels)
        max_count = max(counter.values())
        new_samples, new_labels = [], []
        for cls_idx in range(len(CLASSES)):
            cls_samples = [s for s, l in zip(self.samples, self.labels) if l == cls_idx]
            cls_labels = [cls_idx] * len(cls_samples)
            if len(cls_samples) < max_count:
                cls_samples, cls_labels = resample(cls_samples, cls_labels, replace=True, n_samples=max_count, random_state=42)
            new_samples.extend(cls_samples)
            new_labels.extend(cls_labels)
        self.samples, self.labels = new_samples, new_labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Diagnostic: report which sample index is being loaded
        sample_path = self.samples[idx]
        # Print only occasionally to avoid huge logs
        if idx % 200 == 0:
            print(f"[DATASET] __getitem__ loading idx={idx} path={sample_path}")
        img_tensor = torch.load(sample_path)
        label = self.labels[idx]
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor, label
