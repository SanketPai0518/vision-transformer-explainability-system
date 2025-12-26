import torch
import numpy as np

def attention_rollout(model, x):
    attns = []

    def hook(m, i, o):
        attns.append(o)

    hooks = []
    for blk in model.blocks:
        hooks.append(blk.attn.attn_drop.register_forward_hook(hook))

    _ = model(x)

    for h in hooks:
        h.remove()

    rollout = torch.eye(attns[0].size(-1), device=x.device)
    for attn in attns:
        attn = attn.mean(dim=1)
        rollout = attn @ rollout

    mask = rollout[0, 0, 1:]
    size = int(mask.numel() ** 0.5)
    mask = mask.reshape(size, size)
    mask = mask.detach().cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask
