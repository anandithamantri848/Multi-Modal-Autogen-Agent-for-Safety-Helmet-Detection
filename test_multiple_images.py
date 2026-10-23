from pathlib import Path
from ultralytics import YOLO

model = YOLO("models/best.pt")
image_folder = Path("test_multiple_imgs")

image_files = [
    path for path in image_folder.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
]

if not image_files:
    print("Add traffic images to the traffic_test folder.")
    raise SystemExit

for image_path in image_files:
    print(f"\nTesting: {image_path.name}")

    results = model.predict(
        source=str(image_path),
        conf=0.10,
        save=True,
        verbose=False,
    )

    for result in results:
        counts = {"With Helmet": 0, "Without Helmet": 0}

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            counts[class_name] = counts.get(class_name, 0) + 1

        print("With Helmet:", counts["With Helmet"])
        print("Without Helmet:", counts["Without Helmet"])

print("\nTesting complete. Check the runs/detect folder.")

