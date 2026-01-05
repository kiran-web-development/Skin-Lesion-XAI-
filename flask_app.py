

import os
import sys
import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from lime import lime_image
from skimage.segmentation import mark_boundaries
import io
import base64
from torchvision import transforms
from PIL import ImageDraw, ImageFont

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import models_dir, CLASSES, device, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, lime_dir
from model import get_resnet18_model
from utils import ensure_dir

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}

ensure_dir(UPLOAD_FOLDER)
ensure_dir(lime_dir)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global model and preprocessor
model = None
preprocess = None

def load_model():
    """Load the trained model"""
    global model
    model = get_resnet18_model(pretrained=False)
    model.load_state_dict(torch.load(os.path.join(models_dir, 'best_model.pt'), map_location=device))
    model = model.to(device)
    model.eval()

def get_preprocess():
    """Get preprocessing pipeline"""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(img):
    """Predict class for image"""
    global preprocess
    if preprocess is None:
        preprocess = get_preprocess()
    
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = np.argmax(prob)
    return pred, prob

def explain_with_lime(img_np):
  
    global preprocess
    if preprocess is None:
        preprocess = get_preprocess()
    
    def batch_predict(images):
        
        batch = []
        for im in images:
            im_pil = Image.fromarray(im.astype(np.uint8))
            batch.append(preprocess(im_pil).unsqueeze(0))
        batch = torch.cat(batch).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_np, 
        classifier_fn=batch_predict, 
        top_labels=1, 
        hide_color=0, 
        num_samples=1000
    )
    
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=8, 
        hide_rest=False
    )
    
    # Create image with PIL and convert to base64
    marked_image = mark_boundaries(temp, mask)
    marked_image = (marked_image * 255).astype(np.uint8)
    img_pil = Image.fromarray(marked_image)
    
    # Save to bytes buffer
    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    return img_base64

@app.route('/')
def index():
   
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
        
        # Read and process image
        img = Image.open(file.stream).convert('RGB')
        img_np = np.array(img)
        
        # Predict
        pred_class, probabilities = predict(img)
        pred_name = CLASSES[pred_class]
        confidence = float(probabilities[pred_class]) * 100
        
        # Generate LIME explanation
        lime_img_b64 = explain_with_lime(img_np)
        
        # Prepare results
        probs_dict = {CLASSES[i]: float(probabilities[i]) * 100 for i in range(len(CLASSES))}
        probs_sorted = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Convert input image to base64
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        input_img_b64 = base64.b64encode(buf.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'predicted_class': pred_name,
            'confidence': confidence,
            'all_probabilities': probs_sorted,
            'input_image': f'data:image/png;base64,{input_img_b64}',
            'lime_explanation': f'data:image/png;base64,{lime_img_b64}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    
    return jsonify({'status': 'ok', 'model': 'loaded' if model is not None else 'not_loaded'})

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    print("Model loaded successfully!")
    print("Starting Flask app on http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
