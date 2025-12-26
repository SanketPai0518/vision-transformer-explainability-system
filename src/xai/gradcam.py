import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer
        self.grad = None
        self.act = None

        layer.register_forward_hook(self._forward_hook)
        layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, _, __, output):
        self.act = output

    def _backward_hook(self, _, grad_in, grad_out):
        self.grad = grad_out[0]

    def generate(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        logits[:, class_idx].backward()

        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = (w * self.act).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
