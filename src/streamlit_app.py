import streamlit as st
import torch
import timm
import numpy as np
import os
import csv
from PIL import Image
from torchvision import transforms
from xai.gradcam import GradCAM
from xai.attention_rollout import attention_rollout
from xai.patch_importance import patch_importance
from xai.occlusion import occlusion_sensitivity

# ---------------- config ----------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_vit.pt")

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

os.makedirs(os.path.join(BASE_DIR, "feedback_images"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "feedback"), exist_ok=True)

# ---------------- load model ----------------

@st.cache_resource
def load_model():
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=10
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ---------------- transforms ----------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- UI ----------------

st.title("Vision Transformer Explainability System")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()

    st.subheader("Prediction")
    st.write(f"{CLASSES[pred_idx]} ({probs[0, pred_idx]:.2f})")

    method = st.selectbox(
        "Choose explanation method",
        [
            "Grad-CAM",
            "Attention Rollout",
            "Patch Importance",
            "Occlusion Sensitivity",
            "Top-K Confidence"
        ]
    )

    if method == "Grad-CAM":
        cam = GradCAM(model, model.blocks[-1].norm1)
        heatmap = cam.generate(img_tensor, pred_idx)
        st.image(heatmap, caption="Grad-CAM", use_column_width=True)

    elif method == "Attention Rollout":
        mask = attention_rollout(model, img_tensor)
        st.image(mask, caption="Attention Rollout", use_column_width=True)

    elif method == "Patch Importance":
        patch_map = patch_importance(model, img_tensor)
        st.image(patch_map, caption="Patch Importance", use_column_width=True)

    elif method == "Occlusion Sensitivity":
        patch = st.slider("Patch size", 16, 64, 32)
        occ = occlusion_sensitivity(model, image, transform, patch, DEVICE)
        st.image(occ, caption="Occlusion Sensitivity", use_column_width=True)

    elif method == "Top-K Confidence":
        topk = torch.topk(probs[0], k=5)
        for idx, val in zip(topk.indices, topk.values):
            st.write(f"{CLASSES[idx]}: {val:.2f}")
