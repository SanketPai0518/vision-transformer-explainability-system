from torch import datasets, transforms

transforms = transforms.Compose([ToTensor()])
train_data =datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform)
testset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
    
)