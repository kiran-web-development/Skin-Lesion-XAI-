# app.py: Streamlit demo for skin lesion classification with LIME

import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
from config import models_dir, CLASSES, device, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, lime_dir
from model import get_resnet18_model
from torchvision import transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from utils import ensure_dir

st.title('Skin Lesion Classification with Explainable AI (LIME)')

@st.cache_resource
def load_model():
    model = get_resnet18_model(pretrained=False)
    model.load_state_dict(torch.load(os.path.join(models_dir, 'best_model.pt'), map_location=device))
    model = model.to(device)
    model.eval()
    return model

def get_preprocess():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def predict(img):
    preprocess = get_preprocess()
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = np.argmax(prob)
    return pred, prob

def explain(img):
    explainer = lime_image.LimeImageExplainer()
    img_np = np.array(img)
    def batch_predict(images):
        batch = []
        for im in images:
            im = Image.fromarray(im.astype(np.uint8))
            batch.append(get_preprocess()(im).unsqueeze(0))
        batch = torch.cat(batch).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
    explanation = explainer.explain_instance(
        img_np, classifier_fn=batch_predict, top_labels=1, hide_color=0, num_samples=1000
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=8, hide_rest=False
    )
    fig, ax = plt.subplots()
    ax.imshow(mark_boundaries(temp, mask))
    ax.axis('off')
    st.pyplot(fig)

model = load_model()

uploaded_file = st.file_uploader('Upload a dermoscopic image', type=['jpg', 'jpeg', 'png'])
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Input Image', use_column_width=True)
    pred, prob = predict(img)
    st.write(f'**Predicted Class:** {CLASSES[pred]}')
    st.write(f'**Confidence:** {prob[pred]*100:.2f}%')
    st.write('**LIME Explanation:**')
    explain(img)
