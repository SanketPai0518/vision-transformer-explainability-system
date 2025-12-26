import torch
import numpy as np
import cv2

def attention_rollout(model, img_tensor):
    model.eval()
    attn_weights = []

    def new_forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn_weights.append(attn.detach())

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

    # monkey-patch
    for blk in model.blocks:
        blk.attn.forward = new_forward.__get__(blk.attn, blk.attn.__class__)

    with torch.no_grad():
        _ = model(img_tensor)

    if len(attn_weights) == 0:
        raise RuntimeError("Attention weights not captured")

    # rollout
    rollout = torch.eye(attn_weights[0].size(-1), device=img_tensor.device)

    for attn in attn_weights:
        attn = attn.mean(dim=1)
        attn = attn + torch.eye(attn.size(-1), device=img_tensor.device)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rollout = attn @ rollout

    mask = rollout[0, 0, 1:]

    size = int(mask.numel() ** 0.5)
    mask = mask.reshape(size, size)

    mask = mask.cpu().numpy()
    mask = cv2.resize(mask, (224, 224))
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    return mask
