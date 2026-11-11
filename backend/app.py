
from pathlib import Path
import sys
import uuid

import cv2
import numpy as np
import torch
import re
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from werkzeug.exceptions import RequestEntityTooLarge

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "model" / "best_model.pth"
SRC_DIR = PROJECT_DIR / "src"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_old_gradcam_results(keep=10):
    """Keep the newest 10 Grad-CAM result pairs."""
    result_groups = {}

    for filename in RESULTS_DIR.iterdir():
        if not filename.is_file():
            continue

        match = re.fullmatch(
            r"grad_cam_(heatmap|overlay)_([0-9a-f]{32})\.jpg",
            filename.name,
        )

        if match is None:
            continue

        image_type, image_id = match.groups()
        result_groups.setdefault(image_id, {})[image_type] = filename

    # Sort complete pairs by their newest file modification time.
    complete_groups = []

    for image_id, files in result_groups.items():
        if "heatmap" not in files or "overlay" not in files:
            continue

        newest_time = max(
            files["heatmap"].stat().st_mtime,
            files["overlay"].stat().st_mtime,
        )

        complete_groups.append((newest_time, image_id, files))

    complete_groups.sort(key=lambda item: item[0], reverse=True)

    # Preserve the 10 newest pairs and remove older pairs.
    for _, image_id, files in complete_groups[keep:]:
        for filename in files.values():
            try:
                filename.unlink()
                print(f"Removed old Grad-CAM file: {filename.name}")
            except OSError as exc:
                app.logger.warning(
                    "Could not remove old Grad-CAM file %s: %s",
                    filename.name,
                    exc,
                )


sys.path.insert(0, str(SRC_DIR))
from model import model, device

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

CORS(app, resources={
    r"/*": {
        "origins": [
            "http://127.0.0.1:8787",
            "http://localhost:8787"
        ]
    }
})

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

class_names = checkpoint["class_names"]

DISPLAY_NAMES = {
    "Tomato___Early_blight": "Tomato Early Blight",
    "Tomato___Late_blight": "Tomato Late Blight",
    "Tomato___Leaf_Mold": "Tomato Leaf Mold",
    "Tomato___healthy": "Healthy Tomato Leaf",
}


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(_error):
    return jsonify({
        "success": False,
        "message": "Image exceeds the 10 MB upload limit.",
    }), 413


@app.get("/health")
def health():
    return jsonify({
        "success": True,
        "message": "Crop disease API is running.",
        "model_loaded": True,
        "device": str(device),
        "classes": class_names,
    })


@app.get("/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(RESULTS_DIR, filename)


def generate_grad_cam(input_tensor, class_index, original_image):
    """Generate a Grad-CAM heatmap and overlay."""

    activations = {}
    target_layer = model.layer4[-1]

    def save_activation(_module, _inputs, output):
        # Keep the gradient of the target layer's output.
        if output.requires_grad:
            output.retain_grad()
        activations["value"] = output

    hook = target_layer.register_forward_hook(save_activation)

    try:
        model.zero_grad(set_to_none=True)

        # Ensure the forward pass can calculate gradients,
        # even if the input came from a no-grad prediction pass.
        grad_input = input_tensor.detach().requires_grad_(True)

        with torch.enable_grad():
            outputs = model(grad_input)
            score = outputs[0, class_index]
            score.backward()

        activation = activations.get("value")

        if activation is None or activation.grad is None:
            raise RuntimeError(
                "Grad-CAM activations or gradients are missing."
            )

        gradients = activation.grad
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * activation, dim=1)
        cam = F.relu(cam)
        cam = cam[0].detach().cpu().numpy()
        cam = cam - cam.min()

        maximum = cam.max()
        if maximum > 0:
            cam = cam / maximum

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_JET,
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(
            heatmap,
            (original_image.width, original_image.height),
        )

        original_array = np.array(original_image.convert("RGB"))
        overlay = cv2.addWeighted(
            original_array, 0.6, heatmap, 0.4, 0
        )

        
        image_id = uuid.uuid4().hex
        heatmap_name = f"grad_cam_heatmap_{image_id}.jpg"
        overlay_name = f"grad_cam_overlay_{image_id}.jpg"

        Image.fromarray(heatmap).save(
            RESULTS_DIR / heatmap_name, quality=95
        )
        Image.fromarray(overlay).save(
            RESULTS_DIR / overlay_name, quality=95
        )

        cleanup_old_gradcam_results(keep=10)

        return (
            f"/results/{overlay_name}",
            f"/results/{heatmap_name}",
        )



    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({
            "success": False,
            "message": "No image uploaded. Use the 'file' field.",
        }), 400

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return jsonify({
            "success": False,
            "message": "Please select an image.",
        }), 400

    filename = uploaded_file.filename
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "success": False,
            "message": "Upload a JPG, JPEG, PNG, or WEBP image.",
        }), 400

    try:
        with Image.open(uploaded_file.stream) as opened_image:
            image = opened_image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "success": False,
            "message": "The uploaded file is not a valid image.",
        }), 400

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    try:
        # Normal inference does not need gradients.
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities_tensor = F.softmax(outputs, dim=1)[0]

        probabilities = {
            name: round(float(probabilities_tensor[i].item()), 6)
            for i, name in enumerate(class_names)
        }

        class_index = int(torch.argmax(probabilities_tensor).item())
        predicted_class = class_names[class_index]
        confidence = float(probabilities_tensor[class_index].item())

        overlay_url, heatmap_url = generate_grad_cam(
            input_tensor,
            class_index,
            image,
        )

        return jsonify({
            "success": True,
            "predicted_class": predicted_class,
            "display_name": DISPLAY_NAMES.get(
                predicted_class,
                predicted_class.replace("___", " - ").replace("_", " "),
            ),
            "confidence": confidence,
            "probabilities": probabilities,
            "grad_cam_overlay_url": overlay_url,
            "grad_cam_heatmap_url": heatmap_url,
            "disease_info": None,
        })

    except Exception:
        app.logger.exception("Prediction or Grad-CAM generation failed")
        return jsonify({
            "success": False,
            "message": "Prediction failed. Check the backend terminal for details.",
        }), 500


if __name__ == "__main__":
    print("Crop Disease API is starting...")
    print("Device:", device)
    print("Classes:", class_names)
    app.run(host="127.0.0.1", port=5000, debug=False)
