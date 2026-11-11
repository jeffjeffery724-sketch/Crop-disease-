import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms

from model import model, device


# ==================================================
# 1. Project and Model Paths
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "model" / "best_model.pth"


# ==================================================
# 2. Load Trained Model
# ==================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

class_names = checkpoint["class_names"]

model.eval()


# ==================================================
# 3. Enable Gradients for Grad-CAM
# ==================================================

for parameter in model.layer4.parameters():
    parameter.requires_grad = True


print("Best model loaded successfully!")
print("Classes:", class_names)
print(
    "Validation Accuracy:",
    checkpoint["validation_accuracy"]
)


# ==================================================
# 4. Grad-CAM Activation Hook
# ==================================================

activations = None


def save_activation(module, input, output):

    global activations

    activations = output

    # Keep gradient information
    activations.retain_grad()


target_layer = model.layer4[-1]

target_layer.register_forward_hook(
    save_activation
)


print("Grad-CAM hook registered successfully!")


# ==================================================
# 5. Select Test Image
# ==================================================

IMAGE_PATH = (
    PROJECT_DIR
    / "dataset"
    / "split"
    / "test"
    / "Tomato___Early_blight"
    / "0401adc0-0998-4d8e-90de-4eae8caf4248___RS_Erly.B 8446.JPG"
)


# ==================================================
# 6. Load Image
# ==================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")


# ==================================================
# 7. Image Preprocessing
# ==================================================

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


input_tensor = preprocess(image)

input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(device)


print("Image loaded successfully!")

print(
    "Original image size:",
    image.size
)

print(
    "Input tensor shape:",
    input_tensor.shape
)


# ==================================================
# 8. Model Prediction
# ==================================================

# Enable gradient calculation
with torch.enable_grad():

    outputs = model(input_tensor)


# Convert model output into probabilities

probabilities = F.softmax(
    outputs,
    dim=1
)


# Get highest probability

confidence, predicted_class = torch.max(
    probabilities,
    1
)


predicted_class_index = (
    predicted_class.item()
)

confidence_percentage = (
    confidence.item() * 100
)


# ==================================================
# 9. Display Prediction
# ==================================================

print("\nPrediction")
print("-------------------------")

print(
    "Predicted class:",
    class_names[predicted_class_index]
)

print(
    "Confidence:",
    round(
        confidence_percentage,
        2
    ),
    "%"
)

print(
    "Actual class:",
    "Tomato___Early_blight"
)


# ==================================================
# 10. Calculate Gradients
# ==================================================

model.zero_grad()


# Select score of predicted class

target_score = outputs[
    0,
    predicted_class_index
]


# Backpropagation

target_score.backward()


# Get gradients from activation layer

gradients = activations.grad


# ==================================================
# 11. Check Grad-CAM Data
# ==================================================

print("\nGrad-CAM Information")
print("-------------------------")

print(
    "Gradients captured:",
    gradients is not None
)

print(
    "Activations captured:",
    activations is not None
)


if gradients is not None:

    print(
        "Gradient shape:",
        gradients.shape
    )


if activations is not None:

    print(
        "Activation shape:",
        activations.shape
    )
    
# ==================================================
# 12. Generate Grad-CAM Heatmap
# ==================================================

if gradients is not None and activations is not None:

    # Calculate the importance of each feature map
    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    # Combine feature maps using their weights
    cam = torch.sum(
        weights * activations,
        dim=1
    )

    # Keep only positive values
    cam = F.relu(cam)

    # Convert tensor to NumPy array
    cam = cam.squeeze().detach().cpu().numpy()

    # Normalize heatmap between 0 and 1
    cam = cam - cam.min()

    if cam.max() > 0:
        cam = cam / cam.max()

    # Convert heatmap into a colored image
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    # Convert BGR to RGB
    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    # Resize heatmap to original image dimensions
    heatmap = cv2.resize(
        heatmap,
        (image.width, image.height)
    )

    # Convert original image into NumPy array
    original_image = np.array(image)

    # Blend original image and heatmap
    overlay = cv2.addWeighted(
        original_image,
        0.6,
        heatmap,
        0.4,
        0
    )

    # Create results directory
    RESULTS_DIR = PROJECT_DIR / "results"
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save heatmap and overlay
    heatmap_path = RESULTS_DIR / "grad_cam_heatmap.jpg"
    overlay_path = RESULTS_DIR / "grad_cam_overlay.jpg"

    Image.fromarray(heatmap).save(heatmap_path)
    Image.fromarray(overlay).save(overlay_path)

    print("\nGrad-CAM visualization completed!")
    print("Heatmap saved at:", heatmap_path)
    print("Overlay saved at:", overlay_path)

else:
    print("Grad-CAM failed: gradients or activations are missing.")
