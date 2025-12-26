import streamlit as st
import torch
import timm
import numpy as np
import pandas as pd
from PIL import Image
import os
from torchvision import transforms

# ---------------- config ----------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_vit.pt"

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

os.makedirs("feedback_images", exist_ok=True)
os.makedirs("feedback", exist_ok=True)

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

st.title("Vision Transformer Image Classifier")
st.write("Upload an image and see what the model predicts")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    # preprocess
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    st.subheader("Prediction")
    st.write(f"**Class:** {CLASSES[pred_idx]}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # ---------------- feedback ----------------

    st.divider()
    st.subheader("Is this prediction correct?")

    correct = st.radio(
        "Feedback",
        ["Yes", "No"],
        horizontal=True
    )

    if correct == "No":
        true_label = st.selectbox(
            "Select correct class",
            CLASSES
        )

        if st.button("Submit correction"):
            img_id = len(os.listdir("feedback_images"))

            img_path = f"feedback_images/{img_id}.png"
            image.save(img_path)

            csv_path = "feedback/corrections.csv"

            row = {
                "image_path": img_path,
                "label": CLASSES.index(true_label)
            }

            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            else:
                df = pd.DataFrame([row])

            df.to_csv(csv_path, index=False)

            st.success("Thank you! Feedback saved.")
