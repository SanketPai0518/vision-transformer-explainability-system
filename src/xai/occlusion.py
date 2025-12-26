import torch
import numpy as np
import cv2
from PIL import Image

def occlusion_sensitivity(model, image, transform, patch, device):
    model.eval()

    # baseline prediction
    base_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        base_score = model(base_tensor).max().item()

    img_np = np.array(image)
    h, w, _ = img_np.shape
    heatmap = np.zeros((h, w))

    for y in range(0, h, patch):
        for x in range(0, w, patch):
            occluded = img_np.copy()
            occluded[y:y+patch, x:x+patch] = 0

            # 🔑 convert back to PIL
            occluded_pil = Image.fromarray(occluded)

            t = transform(occluded_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                score = model(t).max().item()

            heatmap[y:y+patch, x:x+patch] = base_score - score

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap
