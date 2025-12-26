import torch
import timm
from torch import nn, optim

from data_ingestion import get_dataloaders
from model_train import train_one_epoch
from model_eval import evaluate
from utils import save_checkpoint
import os
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=True,
        num_classes=10
    ).to(device)

    print("running training...")

    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, "data", "cifar-10-batches-py")

    train_loader, test_loader, classes = get_dataloaders(
        batch_size=32,
        img_size=224,
        data_dir=DATA_DIR
    )


    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(1):

        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)

        print(f"epoch {epoch+1} | loss {loss:.4f} | acc {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            save_checkpoint(model, "best_vit.pt")
            print("saved best model")

