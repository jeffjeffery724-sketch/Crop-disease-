
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

from model import model, device


# 1. Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "model" / "best_model.pth"


# 2. Load the trained model
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
class_names = checkpoint["class_names"]
model.eval()

print("Trained model loaded successfully!")
print("Classes:", class_names)


# 3. Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# 4. Prediction function
def predict_image(image_path):

    image_path = Path(image_path)

    if not image_path.is_file():
        print("Error: Image file not found:", image_path)
        return

    image = Image.open(image_path).convert("RGB")

    input_tensor = preprocess(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)

    confidence, predicted_class = torch.max(
        probabilities,
        dim=1
    )

    class_index = predicted_class.item()
    confidence_percentage = confidence.item() * 100

    print("\nPrediction Result")
    print("-------------------------")
    print("Image:", image_path.name)
    print("Predicted disease:", class_names[class_index])
    print("Confidence:", round(confidence_percentage, 2), "%")


# 5. Test with a leaf image

IMAGE_PATH = (
    PROJECT_DIR
    / "dataset"
    / "split"
    / "test"
    / "Tomato___Leaf_Mold"
    / "02415b3d-2ba8-4654-8398-a480ad901bbf___Crnl_L.Mold 8645.JPG"
)

predict_image(IMAGE_PATH)



