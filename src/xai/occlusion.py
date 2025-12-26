import numpy as np
import torch

def occlusion_sensitivity(model, image, transform, patch, device):
    image = image.resize((224, 224))
    img = np.array(image)
    heatmap = np.zeros((224, 224))

    base = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        base_score = torch.softmax(model(base), 1).max().item()

    for y in range(0, 224, patch):
        for x in range(0, 224, patch):
            occluded = img.copy()
            occluded[y:y+patch, x:x+patch] = 0
            t = transform(occluded).unsqueeze(0).to(device)
            with torch.no_grad():
                score = torch.softmax(model(t), 1).max().item()
            heatmap[y:y+patch, x:x+patch] = base_score - score

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap
