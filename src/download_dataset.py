from datasets import load_dataset

print("Loading PlantVillage dataset...")

dataset = load_dataset(
    "mohanty/PlantVillage",
    revision="refs/convert/parquet"
)

print("Dataset loaded successfully!")
print(dataset)

print("\nTrain dataset:")
print(dataset["train"])

print("\nTest dataset:")
print(dataset["test"])