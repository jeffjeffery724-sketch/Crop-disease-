from pathlib import Path
import json
import random
import shutil
from collections import defaultdict, Counter

# ============================================================
# PATHS
# ============================================================

DATASET_PATH = Path(r".\dataset\plantvillage_raw")
OUTPUT_PATH = Path(r".\dataset\split")

LEAF_MAP_PATH = DATASET_PATH / "leaf_grouping" / "leaf-map.json"

# ============================================================
# SETTINGS
# ============================================================

WANTED_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold"
]

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# ============================================================
# LOAD LEAF MAP
# ============================================================

print("========================================")
print("CREATING LEAKAGE-SAFE DATASET SPLIT")
print("========================================")

print("\nLoading leaf-map.json...")

with open(LEAF_MAP_PATH, "r", encoding="utf-8") as file:
    leaf_map = json.load(file)

print("Leaf-map loaded successfully.")

# ============================================================
# COLLECT IMAGES BY LEAF GROUP
# ============================================================

groups = defaultdict(list)

total_images = 0
matched_images = 0
unmatched_images = 0

print("\nCollecting images and leaf groups...")

for class_name in WANTED_CLASSES:

    class_path = DATASET_PATH / "raw" / "color" / class_name

    if not class_path.exists():
        print(f"ERROR: Folder not found: {class_path}")
        continue

    for image_path in class_path.iterdir():

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        total_images += 1

        filename = image_path.stem

        if "___" not in filename:
            unmatched_images += 1
            continue

        leaf_key = filename.split("___", 1)[1].lower()

        if leaf_key not in leaf_map:
            unmatched_images += 1
            continue

        group_found = False

        for value in leaf_map[leaf_key]:

            parts = value.split(":::")
            mapped_class = parts[0]
            group_id = parts[1]

            if mapped_class == class_name:

                # Group is identified by class + group ID.
                # This prevents group IDs from different classes
                # being mixed together.
                group_key = (class_name, group_id)

                groups[group_key].append(image_path)

                matched_images += 1
                group_found = True
                break

        if not group_found:
            unmatched_images += 1

# ============================================================
# DISPLAY COLLECTION SUMMARY
# ============================================================

print("\n========================================")
print("IMAGE COLLECTION SUMMARY")
print("========================================")

print(f"Total original images : {total_images}")
print(f"Matched images        : {matched_images}")
print(f"Unmatched images      : {unmatched_images}")
print(f"Unique groups         : {len(groups)}")

print("\nGroups per class:")

for class_name in WANTED_CLASSES:

    class_groups = [
        key for key in groups
        if key[0] == class_name
    ]

    class_images = sum(
        len(groups[key])
        for key in class_groups
    )

    print(
        f"{class_name:30} "
        f"Groups: {len(class_groups):3} | "
        f"Images: {class_images:4}"
    )

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

print("\n========================================")
print("PREPARING OUTPUT FOLDERS")
print("========================================")

if OUTPUT_PATH.exists():

    print("Existing split folder found.")
    print("Removing old split folder...")

    shutil.rmtree(OUTPUT_PATH)

for split_name in ["train", "validation", "test"]:

    for class_name in WANTED_CLASSES:

        folder = OUTPUT_PATH / split_name / class_name
        folder.mkdir(parents=True, exist_ok=True)

print("Output folders created.")

# ============================================================
# SPLIT GROUPS
# ============================================================

random.seed(RANDOM_SEED)

split_groups = {
    "train": [],
    "validation": [],
    "test": []
}

print("\n========================================")
print("SPLITTING LEAF GROUPS")
print("========================================")

for class_name in WANTED_CLASSES:

    class_groups = [
        key for key in groups
        if key[0] == class_name
    ]

    random.shuffle(class_groups)

    total_groups = len(class_groups)

    train_count = round(total_groups * TRAIN_RATIO)
    validation_count = round(total_groups * VALIDATION_RATIO)

    # Make sure every group is assigned exactly once.
    test_count = total_groups - train_count - validation_count

    train_groups = class_groups[:train_count]

    validation_groups = class_groups[
        train_count:
        train_count + validation_count
    ]

    test_groups = class_groups[
        train_count + validation_count:
    ]

    split_groups["train"].extend(train_groups)
    split_groups["validation"].extend(validation_groups)
    split_groups["test"].extend(test_groups)

    print(f"\n{class_name}")

    print(f"Total groups      : {total_groups}")
    print(f"Training groups   : {len(train_groups)}")
    print(f"Validation groups : {len(validation_groups)}")
    print(f"Testing groups    : {len(test_groups)}")

# ============================================================
# COPY IMAGES
# ============================================================

print("\n========================================")
print("COPYING IMAGES")
print("========================================")

image_counts = Counter()

for split_name in ["train", "validation", "test"]:

    print(f"\nProcessing {split_name}...")

    for class_name, group_id in split_groups[split_name]:

        group_key = (class_name, group_id)

        images = groups[group_key]

        destination_folder = (
            OUTPUT_PATH
            / split_name
            / class_name
        )

        for image_path in images:

            destination = destination_folder / image_path.name

            shutil.copy2(image_path, destination)

            image_counts[(split_name, class_name)] += 1

print("\nAll matched images copied successfully.")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("FINAL DATASET SPLIT")
print("========================================")

grand_total = 0

for split_name in ["train", "validation", "test"]:

    split_total = 0

    print(f"\n{split_name.upper()}")

    for class_name in WANTED_CLASSES:

        count = image_counts[(split_name, class_name)]

        split_total += count

        print(f"{class_name:30} : {count}")

    print(f"Total {split_name:10} : {split_total}")

    grand_total += split_total

print("\n========================================")
print("TOTAL COPIED IMAGES:", grand_total)
print("========================================")

print("\nOriginal dataset was NOT modified.")
print("No original images were deleted.")
print("Leaf groups were kept together.")
print("========================================")