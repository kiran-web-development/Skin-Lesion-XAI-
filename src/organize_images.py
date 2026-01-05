# organize_images.py: Organize ISIC 2019 images into class folders using ground truth CSV

import os
import shutil
import pandas as pd

# Paths
RAW_DIR = os.path.join(os.path.dirname(__file__), '../data/raw/ISIC2019')
CSV_PATH = os.path.join(RAW_DIR, 'ISIC_2019_Training_GroundTruth (1).csv')
IMG_DIR = os.path.join(RAW_DIR, 'ISIC_2019_Training_Input')


# Mapping: folder name -> CSV column
CLASS_MAP = [
    ('AKIEC', 'AK'),
    ('BCC', 'BCC'),
    ('BKL', 'BKL'),
    ('DF', 'DF'),
    ('MEL', 'MEL'),
    ('NV', 'NV'),
    ('SCC', 'SCC'),
    ('VASC', 'VASC'),
]

def main():
    df = pd.read_csv(CSV_PATH)
    # Remove .jpg from image column if present
    df['image'] = df['image'].astype(str)
    for folder, _ in CLASS_MAP:
        cls_dir = os.path.join(RAW_DIR, folder)
        os.makedirs(cls_dir, exist_ok=True)
    for _, row in df.iterrows():
        img_id = row['image']
        img_file = f"{img_id}.jpg"
        src_path = os.path.join(IMG_DIR, img_file)
        for folder, csv_col in CLASS_MAP:
            # Some CSVs use float (1.0), some int (1), so check both
            if row.get(csv_col, 0) == 1 or row.get(csv_col, 0) == 1.0:
                dst_path = os.path.join(RAW_DIR, folder, img_file)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                break

if __name__ == '__main__':
    main()
