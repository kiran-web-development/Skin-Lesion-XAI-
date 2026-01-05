"""
Test LIME explanation generation
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import io
import base64

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, CLASSES, models_dir, device
from model import get_resnet18_model
from torchvision import transforms

def test_lime_explanation(image_path):
    """Test LIME explanation generation"""
    
    print(f"🎨 Testing LIME Explanation Generation")
    print("=" * 70)
    
    try:
        # Load model
        print("📥 Loading model...")
        model = get_resnet18_model(pretrained=False)
        model.load_state_dict(torch.load(os.path.join(models_dir, 'best_model.pt'), map_location=device))
        model = model.to(device)
        model.eval()
        print(f"✓ Model loaded")
        
        # Load image
        print(f"📸 Loading image...")
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        print(f"✓ Image loaded - Shape: {img_np.shape}")
        
        # Create preprocess function
        preprocess = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        def batch_predict(images):
            """Batch prediction function for LIME"""
            print(f"   [LIME] Processing {len(images)} sample images...")
            batch = []
            for im in images:
                im_pil = Image.fromarray(im.astype(np.uint8))
                batch.append(preprocess(im_pil).unsqueeze(0))
            batch = torch.cat(batch).to(device)
            with torch.no_grad():
                logits = model(batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs
        
        # Generate LIME explanation
        print(f"\n🔄 Generating LIME explanation...")
        print(f"   This may take 30-60 seconds (processing 1000 samples)...")
        
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_np,
            classifier_fn=batch_predict,
            top_labels=1,
            hide_color=0,
            num_samples=1000
        )
        print(f"✓ LIME explanation generated")
        
        # Get image and mask
        print(f"\n📊 Creating explanation visualization...")
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=8,
            hide_rest=False
        )
        print(f"✓ Visualization created")
        print(f"  - Image shape: {temp.shape}")
        print(f"  - Mask shape: {mask.shape}")
        
        # Create PIL image
        print(f"\n🖼️ Converting to PIL image...")
        marked_image = mark_boundaries(temp, mask)
        marked_image = (marked_image * 255).astype(np.uint8)
        img_pil = Image.fromarray(marked_image)
        print(f"✓ PIL image created - Size: {img_pil.size}")
        
        # Convert to base64
        print(f"\n📦 Encoding to base64...")
        buf = io.BytesIO()
        img_pil.save(buf, format='PNG')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        print(f"✓ Base64 encoding successful")
        print(f"  - Encoded size: {len(img_base64)} chars (~{len(img_base64)//1024} KB)")
        
        print(f"\n✅ LIME Explanation test PASSED!")
        print(f"   All components working correctly!")
        
    except Exception as e:
        print(f"\n❌ ERROR during LIME explanation:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    image_path = r"C:\Users\saini\Downloads\original.jpg"
    test_lime_explanation(image_path)
