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
        # output shape: [B, N, D]
        self.act = output

    def _backward_hook(self, _, grad_in, grad_out):
        # grad_out[0] shape: [B, N, D]
        self.grad = grad_out[0]

    def generate(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        logits[:, class_idx].backward()

        # remove CLS token
        act = self.act[:, 1:, :]      # [B, P, D]
        grad = self.grad[:, 1:, :]    # [B, P, D]

        # token importance
        weights = grad.mean(dim=-1)   # [B, P]
        cam = (weights.unsqueeze(-1) * act).sum(dim=-1)

        cam = torch.relu(cam)
        cam = cam[0]

        # reshape patches → grid
        size = int(cam.numel() ** 0.5)
        cam = cam.reshape(size, size)

        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam
