from pathlib import Path
import json
from collections import Counter

DATASET_PATH = Path(r".\dataset\plantvillage_raw")
LEAF_MAP_PATH = DATASET_PATH / "leaf_grouping" / "leaf-map.json"

WANTED_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold"
]

with open(LEAF_MAP_PATH, "r", encoding="utf-8") as file:
    leaf_map = json.load(file)

print("========================================")
print("LEAF GROUP VERIFICATION")
print("========================================")

for class_name in WANTED_CLASSES:

    class_path = DATASET_PATH / "raw" / "color" / class_name

    group_counter = Counter()
    matched_images = 0
    unmatched_images = 0

    for image_path in class_path.iterdir():

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        filename = image_path.stem

        if "___" not in filename:
            unmatched_images += 1
            continue

        leaf_key = filename.split("___", 1)[1].lower()

        if leaf_key not in leaf_map:
            unmatched_images += 1
            continue

        # Find the entry belonging to our current class
        group_found = False

        for value in leaf_map[leaf_key]:

            parts = value.split(":::")
            mapped_class = parts[0]
            group_id = parts[1]

            if mapped_class == class_name:
                group_counter[group_id] += 1
                matched_images += 1
                group_found = True
                break

        if not group_found:
            unmatched_images += 1

    print(f"\n{class_name}")
    print("-" * 50)
    print("Matched images     :", matched_images)
    print("Unmatched images   :", unmatched_images)
    print("Unique leaf groups :", len(group_counter))

    print("\nFirst 10 groups:")

    for group_id, count in list(group_counter.items())[:10]:
        print(f"Group {group_id} -> {count} images")

print("\n========================================")
print("VERIFICATION COMPLETED")
print("No files were moved.")
print("No files were deleted.")
print("========================================")