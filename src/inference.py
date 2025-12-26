import torch
import timm
from data_ingestion import get_dataloaders
from model_eval import evaluate
from utils import load_checkpoint

def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=10
    ).to(device)

    load_checkpoint(model, "best_vit.pt", device)

    _, test_loader, _ = get_dataloaders(
        batch_size=32,
        img_size=224,
        data_dir="data"
    )

    acc = evaluate(model, test_loader, device)
    print("loaded model accuracy:", acc)

if __name__ == "__main__":
    run_inference()
