import torch
import numpy as np
import cv2

def attention_rollout(model, x):
    model.eval()
    B = x.size(0)

    # forward once to build features
    with torch.no_grad():
        _ = model(x)

    rollout = None

    for blk in model.blocks:
        attn = blk.attn

        # --- extract qkv manually ---
        qkv = attn.qkv(attn.norm(x) if hasattr(attn, "norm") else x)
        qkv = qkv.reshape(B, -1, 3, attn.num_heads, attn.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, _ = qkv[0], qkv[1], qkv[2]

        attn_map = (q @ k.transpose(-2, -1)) * attn.scale
        attn_map = attn_map.softmax(dim=-1)

        # average heads
        attn_map = attn_map.mean(dim=1)

        if rollout is None:
            rollout = attn_map
        else:
            rollout = attn_map @ rollout

    # CLS → patch tokens
    mask = rollout[0, 0, 1:]

    size = int(mask.numel() ** 0.5)
    mask = mask.reshape(size, size)

    mask = mask.detach().cpu().numpy()
    mask = cv2.resize(mask, (224, 224))
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    return mask
