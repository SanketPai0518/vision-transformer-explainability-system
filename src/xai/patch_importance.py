import torch
import numpy as np

def patch_importance(model, x):
    attns = []

    def hook(m, i, o):
        attns.append(o)

    hooks = []
    for blk in model.blocks:
        hooks.append(blk.attn.attn_drop.register_forward_hook(hook))

    _ = model(x)

    for h in hooks:
        h.remove()

    attn = torch.stack(attns).mean(0).mean(1)
    scores = attn[0, 1:].mean(0)

    size = int(scores.numel() ** 0.5)
    scores = scores.reshape(size, size)
    scores = scores.detach().cpu().numpy()
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores
