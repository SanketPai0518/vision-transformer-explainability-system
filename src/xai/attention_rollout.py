import torch
import numpy as np

def attention_rollout(model, input_tensor):
    attn_weights = []

    def hook(module, input, output):
        attn_weights.append(output)

    hooks = []
    for blk in model.blocks:
        hooks.append(blk.attn.attn_drop.register_forward_hook(hook))

    _ = model(input_tensor)

    for h in hooks:
        h.remove()

    rollout = torch.eye(attn_weights[0].size(-1)).to(input_tensor.device)

    for attn in attn_weights:
        attn = attn.mean(dim=1)
        rollout = attn @ rollout

    mask = rollout[0, 0, 1:]
    size = int(mask.numel() ** 0.5)
    mask = mask.reshape(size, size)
    mask = mask.detach().cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    return mask
