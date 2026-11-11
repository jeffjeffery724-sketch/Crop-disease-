from pathlib import Path
from PIL import Image

DATASET_PATH = Path(r".\dataset\split")

valid_images = 0
bad_images = 0

print("========================================")
print("IMAGE READABILITY CHECK")
print("========================================")

for image_path in DATASET_PATH.rglob("*"):

    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    try:
        with Image.open(image_path) as img:
            img.verify()

        valid_images += 1

    except Exception as e:
        bad_images += 1
        print("BAD IMAGE:", image_path)
        print("ERROR:", e)

print("\n========================================")
print("RESULT")
print("========================================")
print("Valid images :", valid_images)
print("Bad images   :", bad_images)
print("Total        :", valid_images + bad_images)
print("========================================")