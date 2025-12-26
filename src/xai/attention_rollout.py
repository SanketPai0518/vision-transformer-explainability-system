import torch
import numpy as np
import cv2

def attention_rollout(model, x):
    attn_maps = []

    def hook(module, input, output):
        # output shape: [B, heads, N, N]
        attn_maps.append(output.detach())

    hooks = []
    for blk in model.blocks:
        hooks.append(blk.attn.attn_drop.register_forward_hook(hook))

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    if len(attn_maps) == 0:
        raise RuntimeError("No attention maps captured. Check hook placement.")

    # Average heads
    attn_maps = [a.mean(dim=1) for a in attn_maps]  # [B, N, N]

    # Add identity and normalize
    rollout = torch.eye(attn_maps[0].size(-1), device=x.device)

    for attn in attn_maps:
        attn = attn + torch.eye(attn.size(-1), device=x.device)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rollout = attn @ rollout

    # CLS token → patch tokens
    mask = rollout[0, 0, 1:]

    size = int(mask.numel() ** 0.5)
    mask = mask.reshape(size, size)

    mask = mask.detach().cpu().numpy()
    mask = cv2.resize(mask, (224, 224))
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    return mask
