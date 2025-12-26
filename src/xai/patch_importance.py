import torch
import numpy as np
import cv2

def patch_importance(model, img_tensor):
    model.eval()
    img_tensor.requires_grad_(True)

    logits = model(img_tensor)
    pred = logits.argmax(dim=1)

    logits[:, pred].backward()

    # get patch embeddings
    patches = model.patch_embed(img_tensor)   # [B, P, D]
    grads = img_tensor.grad

    # project gradient importance to patches
    grads = grads.abs().mean(dim=1)  # [B, H, W]
    cam = grads[0]

    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    return cam
