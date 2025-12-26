import torch
import numpy as np

def patch_importance(model, input_tensor):
    attn_maps = []

    def hook(module, input, output):
        attn_maps.append(output)

    hooks = []
    for blk in model.blocks:
        hooks.append(blk.attn.attn_drop.register_forward_hook(hook))

    _ = model(input_tensor)

    for h in hooks:
        h.remove()

    attn = torch.stack(attn_maps).mean(dim=0)
    attn = attn.mean(dim=1)
    patch_scores = attn.mean(dim=1)[0, 1:]

    size = int(patch_scores.numel() ** 0.5)
    patch_scores = patch_scores.reshape(size, size)
    patch_scores = patch_scores.detach().cpu().numpy()
    patch_scores = (patch_scores - patch_scores.min()) / (patch_scores.max() - patch_scores.min() + 1e-8)

    return patch_scores
