import torch
from utils import compute_accuracy

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            c, t = compute_accuracy(logits, y)
            correct += c
            total += t

    return correct / total
