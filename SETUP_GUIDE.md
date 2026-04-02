# Skin Lesion XAI - Setup Guide for Multiple Laptops

This guide will help you set up the Skin Lesion Classification with Explainable AI (LIME) project on any laptop.

## Prerequisites

- **Python 3.8+** (Python 3.10 recommended)
- **Git** (for version control)
- **pip** (Python package installer, comes with Python)
- **CUDA 11.8+** (optional, for GPU acceleration - highly recommended for model training)

### System Requirements
- **RAM**: Minimum 8GB (16GB+ recommended for training)
- **Storage**: 5GB+ free space (for data, models, and outputs)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended for training)

---

## Installation Steps

### Option 1: Automated Setup (Windows)

Run the setup script:
```bash
setup.bat
```

This will:
- Create a virtual environment
- Install all dependencies
- Download and organize data (if configured)
- Verify the setup

### Option 2: Automated Setup (macOS/Linux)

Run the setup script:
```bash
bash setup.sh
```

### Option 3: Manual Setup (All Platforms)

#### 1. **Clone/Extract the Repository**
```bash
# If cloning from Git
git clone <repository-url>
cd skin_lesion_xai

# Or if extracted from ZIP
cd skin_lesion_xai
```

#### 2. **Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. **Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt

# If GPU support is needed (CUDA 11.8)
# Remove torch/torchvision from requirements.txt first, then:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 4. **Verify Installation**
```bash
python verify_setup.py
```

#### 5. **Download Data** (if needed)
```bash
python download_data.py
```

---

## Project Structure After Setup

```
skin_lesion_xai/
├── venv/                          # Virtual environment (created during setup)
├── data/
│   ├── raw/                       # Raw dataset
│   │   └── ISIC2019/
│   └── processed/                 # Preprocessed data
│       ├── train/
│       ├── val/
│       └── test/
├── models/
│   └── best_model.pt             # Trained model (download separately if needed)
├── src/                           # Source code
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain_lime.py
│   └── utils.py
├── lime_outputs/                  # LIME explanations
├── uploads/                       # Web app uploads
├── static/                        # Web assets
├── templates/                     # HTML templates
├── requirements.txt               # Python dependencies
├── setup.bat                      # Windows setup script
├── setup.sh                       # macOS/Linux setup script
├── verify_setup.py                # Setup verification script
├── download_data.py               # Data download script
├── app.py                         # Streamlit demo
├── flask_app.py                   # Flask web app
└── README.md                      # Project documentation
```

---

## Running the Application

### 1. **Activate Virtual Environment**
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. **Run Streamlit Demo**
```bash
streamlit run app.py
```
Opens at: `http://localhost:8501`

### 3. **Run Flask Web App**
```bash
python flask_app.py
```
Opens at: `http://localhost:5000`

### 4. **Train Model** (on ISIC 2019 data)
```bash
python src/train.py
```

### 5. **Evaluate Model**
```bash
python src/evaluate.py
```

### 6. **Generate LIME Explanations**
```bash
python src/explain_lime.py
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError`
**Solution**: Ensure virtual environment is activated and dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: CUDA not found (if using GPU)
**Solution**: Install PyTorch with CUDA support:
```bash
pip uninstall torch torchvision
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Data not found
**Solution**: Download the ISIC 2019 dataset:
```bash
python download_data.py
```

### Issue: Model file not found
**Solution**: Download the pre-trained model or train your own:
```bash
python src/train.py
```

### Issue: Out of memory (GPU/RAM)
**Solution**: 
- Reduce batch size in `src/config.py`
- Use CPU mode: Set `device = 'cpu'` in `src/config.py`
- Reduce image size or number of images

---

## System-Specific Notes

### Windows
- Use `venv\Scripts\activate` instead of `source venv/bin/activate`
- Use `pip install` instead of `pip3 install`

### macOS/Linux
- Use `python3` and `pip3` commands
- May need to install additional system packages for some dependencies

### GPU Setup
For NVIDIA GPUs:
1. Install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Install [cuDNN](https://developer.nvidia.com/cudnn)
3. Install PyTorch with CUDA support (see troubleshooting section)

---

## Additional Resources

- **Dataset**: [ISIC 2019 Challenge](https://challenge2019.isic-archive.com/)
- **Paper**: IEEE Access 2022 - "A Deep Learning Approach Based on Explainable Artificial Intelligence for Skin Lesion Classification"
- **LIME Documentation**: [lime-ml.readthedocs.io](https://lime-ml.readthedocs.io/)
- **PyTorch Documentation**: [pytorch.org](https://pytorch.org/)

---

## Quick Reference Commands

```bash
# Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install/Update dependencies
pip install -r requirements.txt

# Run web demo
streamlit run app.py

# Run Flask app
python flask_app.py

# Verify setup
python verify_setup.py

# Deactivate environment
deactivate
```

---

## Support

For issues or questions, refer to:
1. Check troubleshooting section above
2. Review project README.md
3. Check individual script help: `python <script> --help`

---

**Last Updated**: 2024
**Python Version**: 3.8+
**Status**: Ready for deployment
