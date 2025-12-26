import streamlit as st
import torch
import timm
import os
import hashlib
import csv
import numpy as np
from PIL import Image
from torchvision import transforms

from xai.gradcam import GradCAM
from xai.attention_rollout import attention_rollout
from xai.patch_importance import patch_importance
from xai.occlusion import occlusion_sensitivity
from xai.overlay import overlay_heatmap

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "best_vit.pt")

CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

EXPLANATIONS = {
    "Grad-CAM": "Highlights regions that most influenced the prediction.",
    "Attention Rollout": "Shows how attention flows across transformer layers.",
    "Patch Importance": "Displays important image patches.",
    "Occlusion Sensitivity": "Shows how hiding regions changes confidence.",
    "Top-K Confidence": "Displays uncertainty across classes."
}

@st.cache_resource
def load_model():
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.to(DEVICE).eval()
    return m

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def img_hash(bytes_):
    return hashlib.md5(bytes_).hexdigest()

@st.cache_data

def compute_xai(method, img_bytes, pred, patch):
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    if method == "Grad-CAM":
        cam = GradCAM(model, model.blocks[-1].norm1)
        return cam.generate(img_tensor, pred)

    if method == "Attention Rollout":
        return attention_rollout(model, img_tensor)

    if method == "Patch Importance":
        return patch_importance(model, img_tensor)

    if method == "Occlusion Sensitivity":
        return occlusion_sensitivity(model, image, transform, patch, DEVICE)

st.title("Vision Transformer Explainability System")

uploaded = st.file_uploader("Upload image", ["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input", use_column_width=True)

    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, 1)
        pred = probs.argmax(1).item()

    st.subheader("Prediction")
    st.write(f"{CLASSES[pred]} ({probs[0,pred]:.2f})")

    method = st.selectbox(
        "Explanation method",
        ["Grad-CAM","Attention Rollout","Patch Importance","Occlusion Sensitivity","Top-K Confidence"]
    )

    st.info(EXPLANATIONS[method])

    if method == "Top-K Confidence":
        for i,v in zip(*torch.topk(probs[0],5)):
            st.write(f"{CLASSES[i]}: {v:.2f}")
    else:
        patch = st.slider("Occlusion patch size", 16, 64, 32) if method=="Occlusion Sensitivity" else 32
        key = img_hash(uploaded.getvalue())
        heatmap = compute_xai(method, img_bytes, pred, patch)

        overlay = overlay_heatmap(image, heatmap)
        st.image(overlay, caption="Explanation Overlay", use_column_width=True)

        if method == "Grad-CAM":
            from PIL import Image

        cams = []
        for blk in model.blocks:
            cam = GradCAM(model, blk.norm1)
            cams.append(cam.generate(tensor, pred))

        frames = [Image.fromarray(overlay_heatmap(image, c)) for c in cams]

        gif_path = os.path.join(BASE, "gradcam_layers.gif")
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=400,   # ms per frame
            loop=0
        )

        st.image(gif_path, caption="Grad-CAM Across Layers")


        if st.button("Download Explanation Report"):
            import io, zipfile
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("prediction.txt", f"{CLASSES[pred]} {probs[0,pred]:.2f}")
                Image.fromarray(overlay).save("overlay.png")
                z.write("overlay.png")
            st.download_button("Download ZIP", buf.getvalue(), "xai_report.zip")
