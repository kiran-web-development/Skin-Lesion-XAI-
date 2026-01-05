"""
Direct prediction test without Flask - diagnose model prediction
"""

import sys
import os
import torch
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, CLASSES, models_dir, device
from model import get_resnet18_model
from torchvision import transforms

def test_prediction(image_path):
    """Test model prediction directly"""
    
    print(f"🧠 Testing Model Prediction")
    print("=" * 70)
    
    try:
        # Load model
        print("📥 Loading model...")
        model = get_resnet18_model(pretrained=False)
        model_path = os.path.join(models_dir, 'best_model.pt')
        print(f"   Model path: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ ERROR: Model file not found at {model_path}")
            return
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"✓ Model loaded successfully")
        print(f"  - Device: {device}")
        
        # Load and preprocess image
        print(f"\n📸 Loading image...")
        img = Image.open(image_path).convert('RGB')
        
        preprocess = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        tensor = preprocess(img).unsqueeze(0).to(device)
        print(f"✓ Image loaded and preprocessed")
        print(f"  - Input shape: {tensor.shape}")
        
        # Predict
        print(f"\n🔮 Making prediction...")
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            prediction = np.argmax(probabilities)
        
        print(f"✓ Prediction successful!")
        print(f"\n📊 Results:")
        print(f"  - Predicted Class: {CLASSES[prediction]}")
        print(f"  - Confidence: {probabilities[prediction] * 100:.2f}%")
        
        print(f"\n📋 All Class Probabilities:")
        print("-" * 70)
        for i, (cls_name, prob) in enumerate(zip(CLASSES, probabilities)):
            bar_length = int(prob * 50)
            bar = "█" * bar_length
            print(f"  {i+1}. {cls_name:10s} {prob*100:6.2f}%  {bar}")
        
        print(f"\n✅ Prediction test PASSED!")
        
    except Exception as e:
        print(f"\n❌ ERROR during prediction:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    image_path = r"C:\Users\saini\Downloads\original.jpg"
    test_prediction(image_path)
