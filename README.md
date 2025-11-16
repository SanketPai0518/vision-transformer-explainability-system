Vision Transformer Explainability and Deployment System
----------------------------------------------------------
🚀 Overview
This module brings modern computer vision to life using a Vision Transformer (ViT) combined with explainability tools and a lightweight web deployment API.
Upload an image → Get a prediction → See exactly why the model decided that using Grad-CAM-style attention maps.
Perfect for showing strong ML, CV, and deployment skills.
🧠 Key Features
Train a small ViT on CIFAR/Tiny ImageNet
Export model to ONNX for production
Real-time Grad-CAM visualizations
Fast API or Streamlit deployment
Image gallery for benchmarking predictions
🔧 Tech Stack
PyTorch
timm (Vision Transformer models)
OpenCV, Matplotlib
ONNX Runtime
Streamlit / FastAPI
Docker
🎨 Demo
Launch the local interface:
streamlit run vit_app.py
Then:
Upload an image
View classification
View Grad-CAM overlay
Compare attention heatmaps
📁 Repository Structure
module2_vit_explain/
│── models/
│── notebooks/
│── src/
│   ├── vit_train.py
│   ├── gradcam.py
│   ├── inference.py
│── vit_app.py
│── README.md

🏁 What You Learn / Showcase
Cutting-edge CV
Explainable AI
Model deployment
Efficient inference optimizations