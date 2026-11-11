from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset" / "split"


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Load datasets
train_dataset = datasets.ImageFolder(
    DATASET_DIR / "train",
    transform=transform
)

validation_dataset = datasets.ImageFolder(
    DATASET_DIR / "validation",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    DATASET_DIR / "test",
    transform=transform
)


# Data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# Display information
print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))
print("Testing images:", len(test_dataset))

print("Training batches:", len(train_loader))
print("Validation batches:", len(validation_loader))
print("Testing batches:", len(test_loader))