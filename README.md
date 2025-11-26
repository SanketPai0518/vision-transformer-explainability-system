Vision Transformer Explainability & Deployment System

This module demonstrates a modern computer vision pipeline built around a Vision Transformer (ViT) equipped with explainability and real-time deployment.
Users can upload an image, receive a prediction, and visualize the model’s internal reasoning through Grad-CAM style attention heatmaps.

The project highlights advanced computer vision, transformer architectures, inference optimization, and API/web deployment skills.

🚀 Overview

Transformers have reshaped natural language processing—and now they dominate computer vision as well.
This module implements:

A compact Vision Transformer trained on CIFAR-10 or Tiny ImageNet

Explainability tools to make predictions interpretable

A streamlined deployment interface (Streamlit or FastAPI)

Optional ONNX export for optimized inference

This creates an end-to-end showcase of image classification, attention visualization, and model serving.

🧠 Key Features
✔ Vision Transformer Training

Train a lightweight ViT on CIFAR-10 or Tiny ImageNet

Custom data augmentation pipeline

Early stopping, LR scheduling, and checkpointing

Option to fine-tune pre-trained ViT models via timm

✔ Explainability (Grad-CAM / Attention Maps)

Visualize model attention on input images

Overlay Grad-CAM heatmaps on original image

Inspect token-wise contributions

✔ Fast Inference & Deployment

Export trained model to ONNX

Serve predictions using:

Streamlit UI (interactive demo), or

FastAPI (production-style API endpoint)

Real-time heatmap generation for uploaded images

✔ Benchmarking & Comparison

Image gallery for testing predictions

Compare explanations across multiple samples

Optional: compare CNN vs ViT behavior

🔧 Tech Stack

PyTorch (training & inference)

timm (Vision Transformer architectures)

OpenCV / Matplotlib (visualization)

ONNX Runtime (accelerated inference)

Streamlit or FastAPI (deployment)

Docker (containerization, optional)

🎨 Demo

Launch the interactive UI:

streamlit run vit_app.py


Features include:

Image upload

Top-k predictions

Grad-CAM heatmaps

Attention visualization

Comparison gallery

📁 Repository Structure
module2_vit_explain/
│── models/                 # Saved weights (PyTorch & ONNX)
│── notebooks/              # Training and explainability exploration
│── src/
│   ├── vit_train.py        # Training script for ViT
│   ├── gradcam.py          # Grad-CAM implementation for transformers
│   ├── inference.py        # Inference utilities (torch & ONNX)
│── vit_app.py              # Streamlit UI / deployment app
│── README.md               # This file

🏁 Skills Demonstrated

Modern computer vision using Vision Transformers

Explainable AI with Grad-CAM and attention visualization

Model optimization through ONNX export

Deployment engineering (Streamlit / FastAPI / Docker)

End-to-end ML pipeline design

Interpretable, production-ready image classification

This module becomes the second core piece of your MEGA SAGA AI portfolio.