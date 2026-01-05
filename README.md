# Skin Lesion Classification with Explainable AI (LIME)

This repository implements the IEEE Access 2022 paper:

**“A Deep Learning Approach Based on Explainable Artificial Intelligence for Skin Lesion Classification”**

using ResNet-18 and LIME for the ISIC 2019 dataset.

## Project Structure

```
skin_lesion_xai/
├── data/
│   ├── raw/ISIC2019/
│   ├── processed/
├── models/
├── lime_outputs/
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── augmentation.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain_lime.py
│   └── utils.py
├── app.py (optional demo)
├── requirements.txt
└── README.md
```

## Problem Statement

Automatic classification of skin lesions using deep learning, with post-hoc explainability via LIME, to assist dermatologists in diagnosis.

## Dataset

- **ISIC 2019**: 25,331 dermoscopic images, 8 classes (AKIEC, BCC, BKL, DF, MEL, NV, SCC, VASC)
- Severe class imbalance handled as per the paper

## Methodology

- **Preprocessing**: ROI cropping, zero-padding, resize (224x224), normalization (ImageNet stats)
- **Augmentation**: Random flip, rotation, crop, color jitter
- **Model**: ResNet-18 (ImageNet pretrained), 8-class head, transfer learning, fine-tuning
- **Training**: Adam, lr=0.0001, batch=32, epochs=25-30, weighted loss, early stopping, CUDA
- **Evaluation**: Accuracy, Precision, Recall, F1, Confusion Matrix
- **Explainability**: LIME for superpixel-based explanations

## How to Run

1. Download ISIC 2019 dataset to `data/raw/ISIC2019/`
2. Install requirements: `pip install -r requirements.txt`
3. Preprocess and augment data: `python src/preprocessing.py`
4. Train: `python src/train.py`
5. Evaluate: `python src/evaluate.py`
6. Generate explanations: `python src/explain_lime.py`
7. (Optional) Run demo app: `python app.py`

See each script for more details.

## References
- [IEEE Access Paper](https://ieeexplore.ieee.org/document/9746140)
- [ISIC 2019 Dataset](https://challenge2019.isic-archive.com/)
