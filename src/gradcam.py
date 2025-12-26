import torch
import torch.nn.functional as F
import numpy as np
import cv2


class ViTGradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        # hook last transformer block output
        block = self.model.blocks[-1]

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        block.register_forward_hook(forward_hook)
        block.register_full_backward_hook(backward_hook)

    def generate(self, x, class_idx=None):
        x = x.requires_grad_(True)

        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1)

        score = logits[0, class_idx]
        self.model.zero_grad()
        score.backward()

        # remove CLS token
        activations = self.activations[:, 1:, :]
        gradients = self.gradients[:, 1:, :]

        # global average pooling over tokens
        weights = gradients.mean(dim=1, keepdim=True)

        cam = (weights * activations).sum(dim=-1)
        cam = F.relu(cam)

        cam = cam.reshape(14, 14)
        cam = cam.detach().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def overlay_cam(img, cam):
    h, w = img.shape[:2]

    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    overlay = heatmap * 0.4 + img * 255
    overlay = overlay / 255.0

    return overlay
