import torch
import tqdm
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    from tqdm import tqdm

    for x, y in tqdm(loader, desc="training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()



        total_loss += loss.item()

    return total_loss / len(loader)
