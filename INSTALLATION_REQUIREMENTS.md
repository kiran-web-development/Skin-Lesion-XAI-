# Installation Requirements - Skin Lesion XAI

Complete breakdown of system and software requirements for deploying this project on other laptops.

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Ubuntu 18.04+
- **RAM**: 8 GB
- **Storage**: 5 GB free space (for code, models, and outputs)
- **CPU**: Intel/AMD multi-core processor (2+ cores)
- **Internet**: Required for initial setup and data download

### Recommended Requirements (for Training)
- **OS**: Windows 11, macOS 12+, or Ubuntu 20.04+
- **RAM**: 16 GB or more
- **Storage**: 50+ GB (for full ISIC 2019 dataset)
- **CPU**: Intel i7/i9 or AMD Ryzen 7/9 (6+ cores)
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- **Internet**: Stable connection for long training sessions

## Software Requirements

### Python Environment
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **Package Manager**: pip (comes with Python)
- **Virtual Environment**: venv (included with Python 3.3+)

### Core Dependencies
All dependencies are listed in `requirements.txt`:

```
torch>=1.10.0                 # Deep learning framework
torchvision>=0.11.0           # Computer vision utilities
scikit-learn>=1.0.0           # Machine learning utilities
numpy>=1.21.0                 # Numerical computing
pandas>=1.3.0                 # Data manipulation
matplotlib>=3.4.0             # Plotting library
opencv-python>=4.5.0          # Image processing
Pillow>=8.0.0                 # Image library
lime>=0.2.0.1                 # Explainable AI
tqdm>=4.62.0                  # Progress bars
seaborn>=0.11.0               # Statistical plotting
scipy>=1.7.0                  # Scientific computing
streamlit>=1.0.0              # Web interface (optional)
flask>=2.0.0                  # Web framework
```

### Optional Dependencies

#### For GPU Support (CUDA)
- **NVIDIA CUDA Toolkit**: 11.8+ (for GPU acceleration)
- **NVIDIA cuDNN**: 8.0+ (for deep learning GPU support)
- **NVIDIA Driver**: Latest stable version
- **PyTorch GPU Support**: Install via `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

#### For Development
- **Git**: 2.0+ (for version control)
- **Visual Studio Code** or IDE of choice
- **Jupyter Notebook**: For interactive analysis (optional)

## Platform-Specific Requirements

### Windows 10/11

**Base Installation Steps**:
1. Download Python 3.10+ from [python.org](https://www.python.org/)
2. Run installer and check "Add Python to PATH"
3. Verify: Open Command Prompt and run `python --version`

**For GPU Support**:
1. Download NVIDIA CUDA Toolkit 11.8 from [nvidia.com](https://developer.nvidia.com/cuda-downloads)
2. Download cuDNN from [nvidia.com](https://developer.nvidia.com/cudnn)
3. Follow NVIDIA installation guide for Windows

**Additional Tools**:
- Visual C++ Build Tools (may be required for some packages)
- Windows Terminal (optional but recommended)

### macOS

**Base Installation Steps**:
1. Install Python 3.10+ via [Homebrew](https://brew.sh/):
   ```bash
   brew install python@3.10
   ```
2. Verify: Open Terminal and run `python3 --version`

**For GPU Support**:
- Note: Most Macs use Apple Silicon (M1/M2) or Intel processors
- For Apple Silicon: Use `pytorch-nightly` for better support
- For Intel Macs: Standard CUDA support similar to Linux

**Additional Tools**:
- Xcode Command Line Tools: `xcode-select --install`
- Homebrew (package manager)

### Linux (Ubuntu 18.04+)

**Base Installation Steps**:
1. Update system:
   ```bash
   sudo apt update && sudo apt upgrade
   ```
2. Install Python:
   ```bash
   sudo apt install python3.10 python3.10-venv python3-pip
   ```
3. Verify: `python3 --version`

**For GPU Support**:
1. Install NVIDIA Container Toolkit (if using Docker)
2. Or install CUDA Toolkit and cuDNN separately

**Additional Tools**:
- Build essentials: `sudo apt install build-essential`
- Git: `sudo apt install git`

## Network Requirements

### Initial Setup
- **Bandwidth**: 500 MB - 2 GB (for pip packages)
- **Time**: 15-30 minutes on typical internet connection

### Data Download (If Needed)
- **ISIC 2019 Dataset**: 50+ GB
- **Bandwidth**: Minimum 10 Mbps recommended
- **Time**: 1-2 hours depending on connection

## Disk Space Breakdown

```
Minimal Installation:
├── Python + Dependencies: ~2-3 GB
├── Project Code: ~100 MB
├── Models (pre-trained): ~100-200 MB
└── Temp/Cache: ~500 MB
Total: ~3-4 GB

With Full ISIC 2019 Dataset:
├── Python + Dependencies: ~2-3 GB
├── Project Code: ~100 MB
├── Raw Dataset: ~50+ GB
├── Processed Dataset: ~20+ GB
├── Models: ~200 MB
└── LIME Outputs: ~5+ GB
Total: ~75-80 GB
```

## GPU Requirements Detail

### NVIDIA GPU Compatibility
**Supported GPUs**:
- NVIDIA Tesla series (T4, A10, A100, etc.)
- NVIDIA GeForce RTX series (RTX 3060, 3070, 3080, 4070, 4080, 4090, etc.)
- NVIDIA GeForce GTX series (GTX 1060, 1080, 1080 Ti, etc.)
- NVIDIA Quadro series

**Minimum VRAM**:
- Inference only: 2 GB
- Training: 4-6 GB
- Large batch training: 8+ GB

**Recommended Setup**:
- GPU: NVIDIA RTX 3060 Ti or better
- VRAM: 8-12 GB
- CUDA Version: 11.8
- cuDNN Version: 8.0+

### Checking GPU Compatibility
```bash
# Check CUDA capability
python -c "import torch; print(torch.cuda.is_available())"

# List available GPUs
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"
```

## Bandwidth & Firewall

### Required Network Access
- **PyPI (pip)**: For package installation
- **GitHub**: For source code (if cloning)
- **ISIC Dataset Server**: For dataset download
- **Model Hub**: If downloading pre-trained models

### Firewall Ports
- **8501**: Streamlit app (default)
- **5000**: Flask app (default)
- **443/80**: Standard HTTP/HTTPS for downloads

## Verification Commands

### Python
```bash
python --version           # Check Python version
pip --version             # Check pip version
python -m venv --help     # Verify venv available
```

### Dependencies
```bash
pip list                  # List installed packages
python verify_setup.py    # Run project verification script
```

### GPU
```bash
nvidia-smi                # Check NVIDIA driver and GPUs
python -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting Installation Issues

### Common Issues

#### 1. Python Not Found
**Solution**: Ensure Python is in PATH or use full path to Python executable

#### 2. pip Not Available
**Solution**: 
```bash
python -m pip install --upgrade pip
```

#### 3. Virtual Environment Fails
**Solution**:
```bash
python -m venv venv --clear   # Clear corrupted venv
python -m venv venv            # Recreate
```

#### 4. CUDA Not Found
**Solution**: Install CUDA Toolkit and cuDNN from NVIDIA website

#### 5. Out of Memory
**Solution**: 
- Reduce batch size in config
- Use CPU mode
- Upgrade RAM or use external storage

#### 6. Slow Installation
**Solution**:
- Check internet connection
- Use `-U` flag to skip building wheels: `pip install -r requirements.txt -U`
- Consider using pre-built wheels for your platform

## Post-Installation Verification

Run the verification script after installation:
```bash
source venv/bin/activate        # macOS/Linux
# or
venv\Scripts\activate           # Windows

python verify_setup.py
```

## Support Resources

- **Python**: [python.org](https://python.org)
- **PyTorch**: [pytorch.org](https://pytorch.org)
- **CUDA**: [developer.nvidia.com](https://developer.nvidia.com/cuda-toolkit)
- **Project Issues**: Check project README or issue tracker

---

**Last Updated**: 2024
**Compatible Python Versions**: 3.8, 3.9, 3.10, 3.11
**PyTorch Version**: 1.10.0+
