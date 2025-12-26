import torch
import numpy as np
import cv2

def occlusion_sensitivity(model, image, transform, patch_size, device):
    model.eval()
    image_np = np.array(image.resize((224, 224)))
    heatmap = np.zeros((224, 224))

    base_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        base_prob = torch.softmax(model(base_tensor), dim=1).max().item()

    for y in range(0, 224, patch_size):
        for x in range(0, 224, patch_size):
            occluded = image_np.copy()
            occluded[y:y+patch_size, x:x+patch_size] = 0

            tensor = transform(cv2.cvtColor(occluded, cv2.COLOR_RGB2BGR))
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                prob = torch.softmax(model(tensor), dim=1).max().item()

            heatmap[y:y+patch_size, x:x+patch_size] = base_prob - prob

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap
