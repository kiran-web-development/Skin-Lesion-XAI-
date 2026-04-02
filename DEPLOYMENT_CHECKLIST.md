# Deployment Checklist - Skin Lesion XAI

Use this checklist when deploying the Skin Lesion Classification project to a new laptop.

## Pre-Deployment

- [ ] Verify you have the complete project folder structure
- [ ] Check that all source code files are present (see SETUP_GUIDE.md)
- [ ] Ensure the requirements.txt file is up to date
- [ ] Create a backup of the project before starting

## Environment Setup

### Python & Virtual Environment
- [ ] Python 3.8+ is installed
- [ ] pip is available and working
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Activate environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
- [ ] Virtual environment is active (check prompt shows `(venv)`)

### Dependencies Installation
- [ ] Upgrade pip: `python -m pip install --upgrade pip setuptools wheel`
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Run verification: `python verify_setup.py`
- [ ] All checks passed (or documented why not)

## GPU/CUDA Setup (If Applicable)

- [ ] NVIDIA CUDA Toolkit installed (if using GPU)
- [ ] cuDNN installed (if using GPU)
- [ ] PyTorch GPU support verified: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] CUDA version matches PyTorch requirements

## Data & Models

- [ ] `data/` directory exists
- [ ] `data/raw/` directory created
- [ ] `data/processed/` directory created
- [ ] `models/` directory exists
- [ ] Pre-trained model available at `models/best_model.pt` (if needed)
- [ ] Test images available in `test_images/` folder
- [ ] Data preprocessing complete (if raw data is available)

## Project Configuration

- [ ] `src/config.py` reviewed and updated if needed
- [ ] Device settings correct (GPU/CPU choice in config)
- [ ] Data paths are correct
- [ ] Model paths are correct
- [ ] Image size settings appropriate for your use case

## Application Testing

### Streamlit Demo
- [ ] Run: `streamlit run app.py`
- [ ] Web interface loads at `http://localhost:8501`
- [ ] Upload image functionality works
- [ ] Model predictions work
- [ ] LIME explanations generate successfully
- [ ] Stop: Press `Ctrl+C` in terminal

### Flask Web App
- [ ] Run: `python flask_app.py`
- [ ] Web interface loads at `http://localhost:5000`
- [ ] File upload works
- [ ] Predictions display correctly
- [ ] Explanations render properly
- [ ] Stop: Press `Ctrl+C` in terminal

### Training & Evaluation (If Needed)
- [ ] Full ISIC 2019 dataset available (if training)
- [ ] Run training: `python src/train.py`
- [ ] Model training completes without errors
- [ ] Evaluation script works: `python src/evaluate.py`
- [ ] LIME explanations generate: `python src/explain_lime.py`

## Troubleshooting & Issues

- [ ] Document any errors encountered
- [ ] Refer to SETUP_GUIDE.md troubleshooting section
- [ ] Check compatibility issues with this system
- [ ] Verify all dependencies are compatible versions
- [ ] Test on sample data first

## Performance Validation

- [ ] Initial model load time acceptable
- [ ] Inference speed satisfactory (GPU/CPU as appropriate)
- [ ] Memory usage acceptable
- [ ] No crashes or errors during normal operation
- [ ] LIME explanation generation time acceptable

## Documentation & Handoff

- [ ] SETUP_GUIDE.md reviewed
- [ ] Project README.md reviewed
- [ ] System specifications documented
- [ ] Known issues documented
- [ ] Next steps for user documented
- [ ] Contact information provided

## Post-Setup

- [ ] Create system snapshot/backup (optional but recommended)
- [ ] Document custom configurations made
- [ ] Test on fresh system if possible
- [ ] Create shortcut/launcher for easy access (optional)
- [ ] Set up auto-startup if needed (production deployment)

## System Information to Document

```
Device Information:
- Computer Name: ________________
- OS: ________________
- Python Version: ________________
- CUDA Version (if GPU): ________________
- GPU Model (if applicable): ________________
- RAM: ________________
- Storage: ________________

Setup Date: ________________
Setup By: ________________
Last Verified: ________________
Notes: 
_________________________________________________________________
_________________________________________________________________
```

## Sign-Off

- [ ] Ready for use
- [ ] All tests passed
- [ ] User trained on usage
- [ ] Documentation provided

**Deployment Date**: __________________

**Deployed By**: __________________

**Verified By**: __________________

---

## Quick Reference

### Common Commands
```bash
# Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Run Streamlit
streamlit run app.py

# Run Flask
python flask_app.py

# Run verification
python verify_setup.py

# Deactivate environment
deactivate
```

### Directory Structure
```
skin_lesion_xai/
├── venv/
├── data/
├── models/
├── src/
├── static/
├── templates/
├── requirements.txt
├── SETUP_GUIDE.md
├── verify_setup.py
└── app.py
```

### Troubleshooting Commands
```bash
# Check Python
python --version

# Check pip
pip list

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Upgrade pip
python -m pip install --upgrade pip
```

---

**For detailed setup instructions, see SETUP_GUIDE.md**
