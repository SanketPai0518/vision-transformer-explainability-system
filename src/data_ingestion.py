import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []

        if train:
            files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            files = ["test_batch"]

        for file in files:
            path = os.path.join(data_dir, file)
            with open(path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
                self.data.append(batch[b"data"])
                self.labels.extend(batch[b"labels"])

        self.data = np.concatenate(self.data)
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        img = torch.tensor(img, dtype=torch.float32) / 255.0

        if self.transform:
            img = self.transform(img)

        return img, label


def get_dataloaders(batch_size, img_size, data_dir):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size))
    ])

    train_ds = CIFAR10Dataset(
        data_dir=data_dir,
        train=True,
        transform=transform
    )

    test_ds = CIFAR10Dataset(
        data_dir=data_dir,
        train=False,
        transform=transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    return train_loader, test_loader, classes
