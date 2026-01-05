# ISIC 2019 split generator (run once after preprocessing)
import os
import random
from config import processed_data_dir, CLASSES
from utils import set_seed

set_seed(42)

train_ratio = 0.7
val_ratio = 0.15
# test_ratio = 0.15 (not used in this implementation)

train_lines, val_lines = [], []

for cls in CLASSES:
    cls_dir = os.path.join(processed_data_dir, cls)
    files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith('.pt')]
    random.shuffle(files)
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    train_lines += [f+','+cls+'\n' for f in train_files]
    val_lines += [f+','+cls+'\n' for f in val_files]

with open(os.path.join(processed_data_dir, 'train_split.csv'), 'w') as f:
    f.writelines(train_lines)
with open(os.path.join(processed_data_dir, 'val_split.csv'), 'w') as f:
    f.writelines(val_lines)
