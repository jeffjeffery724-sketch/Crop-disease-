import torch
import torch.nn as nn
from torchvision import models


# Number of disease classes
NUM_CLASSES = 4


# Load pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)


# Freeze the pretrained layers
for parameter in model.parameters():
    parameter.requires_grad = False


# Replace the final classification layer
number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    NUM_CLASSES
)


# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)


# Display model information
print("Model: ResNet18")
print("Number of classes:", NUM_CLASSES)
print("Device:", device)
print("Final layer:", model.fc)