import torch
import torch.nn as nn
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from data_loader import test_loader
from model import model, device


# -------------------------
# Load saved model
# -------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "model" / "best_model.pth"

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

class_names = checkpoint["class_names"]

print("Loaded model successfully!")
print("Classes:", class_names)
print(
    "Best validation accuracy:",
    checkpoint["validation_accuracy"]
)


# -------------------------
# Evaluation
# -------------------------

model.eval()

criterion = nn.CrossEntropyLoss()

test_loss = 0.0
correct = 0
total = 0

all_predictions = []
all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        # Store predictions
        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# -------------------------
# Accuracy
# -------------------------

test_loss = test_loss / len(test_loader)

test_accuracy = 100 * correct / total


print("\nTest Results")
print("-------------------------")

print(
    "Test Loss:",
    round(test_loss, 4)
)

print(
    "Test Accuracy:",
    round(test_accuracy, 2),
    "%"
)

print(
    "Correct Predictions:",
    correct,
    "/",
    total
)


# -------------------------
# Classification Report
# -------------------------

print("\nClassification Report")
print("-------------------------")

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=4
)

print(report)


# -------------------------
# Confusion Matrix
# -------------------------

print("\nConfusion Matrix")
print("-------------------------")

matrix = confusion_matrix(
    all_labels,
    all_predictions
)

print(matrix)