#!/usr/bin/env python3
"""
Setup Verification Script for Skin Lesion XAI

This script verifies that all required dependencies and configurations are
correctly set up for the Skin Lesion Classification project.

Usage:
    python verify_setup.py
"""

import sys
import os
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header."""
    print(f"\n{BOLD}{BLUE}{'='*50}{RESET}")
    print(f"{BOLD}{BLUE}{text:^50}{RESET}")
    print(f"{BOLD}{BLUE}{'='*50}{RESET}\n")


def print_success(text):
    """Print a success message."""
    print(f"{GREEN}✓ {text}{RESET}")


def print_error(text):
    """Print an error message."""
    print(f"{RED}✗ {text}{RESET}")


def print_warning(text):
    """Print a warning message."""
    print(f"{YELLOW}⚠ {text}{RESET}")


def print_info(text):
    """Print an info message."""
    print(f"{BLUE}ℹ {text}{RESET}")


def check_python_version():
    """Check Python version."""
    print_info("Checking Python version...")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} (3.8+ required)")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_required_packages():
    """Check if all required packages are installed."""
    print_info("Checking required packages...\n")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'sklearn': 'scikit-learn',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'lime': 'LIME',
        'tqdm': 'tqdm',
        'seaborn': 'Seaborn',
        'scipy': 'SciPy',
        'streamlit': 'Streamlit',
        'flask': 'Flask',
    }
    
    all_installed = True
    installed_packages = []
    missing_packages = []
    
    for package_name, display_name in required_packages.items():
        try:
            __import__(package_name)
            print_success(f"{display_name}")
            installed_packages.append(display_name)
        except ImportError:
            print_error(f"{display_name} - NOT INSTALLED")
            missing_packages.append(display_name)
            all_installed = False
    
    return all_installed, installed_packages, missing_packages


def check_directories():
    """Check if required directories exist or can be created."""
    print_info("Checking directories...\n")
    
    required_dirs = [
        'data',
        'data/raw',
        'data/processed',
        'models',
        'lime_outputs',
        'uploads',
        'src',
        'static',
        'templates',
    ]
    
    all_exist = True
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print_success(f"{directory}")
        else:
            print_warning(f"{directory} - MISSING (will be created if needed)")
            try:
                os.makedirs(directory, exist_ok=True)
                print_info(f"   → Created {directory}")
            except Exception as e:
                print_error(f"   → Cannot create {directory}: {e}")
                all_exist = False
    
    return all_exist


def check_files():
    """Check if critical files exist."""
    print_info("Checking critical files...\n")
    
    critical_files = [
        'requirements.txt',
        'README.md',
        'src/config.py',
        'src/model.py',
        'src/dataset.py',
        'src/train.py',
        'src/evaluate.py',
        'src/explain_lime.py',
        'src/utils.py',
        'app.py',
        'flask_app.py',
    ]
    
    all_exist = True
    
    for file_path in critical_files:
        if os.path.exists(file_path):
            print_success(f"{file_path}")
        else:
            print_warning(f"{file_path} - MISSING")
            all_exist = False
    
    return all_exist


def check_model_file():
    """Check if the pre-trained model exists."""
    print_info("Checking model file...\n")
    
    model_path = 'models/best_model.pt'
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print_success(f"{model_path} ({size_mb:.1f} MB)")
        return True
    else:
        print_warning(f"{model_path} - NOT FOUND")
        print_info("   → You can train your own model using: python src/train.py")
        return False


def check_data():
    """Check if test data exists."""
    print_info("Checking data...\n")
    
    data_path = 'test_images'
    
    if os.path.exists(data_path):
        num_images = len([f for f in os.listdir(data_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        if num_images > 0:
            print_success(f"Test images found ({num_images} images)")
            return True
        else:
            print_warning(f"Test images directory found but empty")
            return False
    else:
        print_warning(f"{data_path} - NOT FOUND")
        print_info("   → You can add test images to this directory")
        return False


def check_gpu():
    """Check if GPU/CUDA is available."""
    print_info("Checking GPU/CUDA support...\n")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print_success(f"GPU support enabled ({gpu_count} GPU(s) found)")
            for i in range(gpu_count):
                device_name = torch.cuda.get_device_name(i)
                print_info(f"   → GPU {i}: {device_name}")
            return True
        else:
            print_warning("GPU/CUDA not available - using CPU mode")
            print_info("   → For faster training, install CUDA and PyTorch with GPU support")
            return False
    except Exception as e:
        print_warning(f"Cannot check GPU: {e}")
        return False


def check_torch_versions():
    """Check PyTorch and related versions."""
    print_info("Checking PyTorch versions...\n")
    
    try:
        import torch
        print_success(f"PyTorch: {torch.__version__}")
        
        import torchvision
        print_success(f"TorchVision: {torchvision.__version__}")
        
        return True
    except Exception as e:
        print_error(f"Cannot check PyTorch versions: {e}")
        return False


def generate_report(results):
    """Generate a summary report."""
    print_header("Setup Verification Report")
    
    total_checks = len(results)
    passed_checks = sum(1 for result in results.values() if result)
    
    status = f"{passed_checks}/{total_checks} checks passed"
    
    if passed_checks == total_checks:
        print_success(status)
        print(f"\n{GREEN}✓ Your setup is complete and ready to use!{RESET}")
        return True
    else:
        print_warning(status)
        failed_checks = [name for name, result in results.items() if not result]
        print(f"\n{YELLOW}The following checks did not pass:{RESET}")
        for check in failed_checks:
            print(f"  • {check}")
        print(f"\n{YELLOW}Please review the issues above and rerun this script.{RESET}")
        return False


def main():
    """Main function."""
    print_header("Skin Lesion XAI - Setup Verification")
    
    results = {}
    
    # Run all checks
    results['Python Version'] = check_python_version()
    
    print_header("Package Dependencies")
    all_packages, installed, missing = check_required_packages()
    results['Required Packages'] = all_packages
    
    if missing:
        print(f"\n{YELLOW}Missing packages: {', '.join(missing)}{RESET}")
        print(f"{BLUE}Install missing packages with: pip install -r requirements.txt{RESET}\n")
    
    print_header("Project Structure")
    results['Directories'] = check_directories()
    
    print_header("Critical Files")
    results['Critical Files'] = check_files()
    
    print_header("Model & Data")
    results['Model File'] = check_model_file()
    results['Test Data'] = check_data()
    
    print_header("GPU & Acceleration")
    check_gpu()
    check_torch_versions()
    
    # Generate report
    report_success = generate_report(results)
    
    print("\n" + "="*50 + "\n")
    
    if report_success:
        print(f"{GREEN}Next Steps:{RESET}")
        print("  1. Activate virtual environment: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
        print("  2. Run demo: streamlit run app.py")
        print("  3. Or run Flask app: python flask_app.py")
        print("  4. For more info, see SETUP_GUIDE.md")
        sys.exit(0)
    else:
        print(f"{YELLOW}Please fix the issues above and rerun this script.{RESET}")
        sys.exit(1)


if __name__ == '__main__':
    main()
