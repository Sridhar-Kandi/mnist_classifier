import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config


def get_mnist_dataloader(batch_size):
    """
    Downloads the MNIST dataset and returns a Dataloader for it.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    train_dataset = datasets.MNIST(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffle the data at every epoch
    )

    test_dataset = datasets.MNIST(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle test data
    )

    print("Data loaders created successfully.")
    return train_loader, test_loader