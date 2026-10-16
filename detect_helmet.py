
from ultralytics import YOLO
from pathlib import Path

# Load the trained helmet detection model
model = YOLO("models/best.pt")

# Display the classes the model recognises
print("Model classes:", model.names)

# Set the traffic image path
image_path = Path("traffic.jpg")

if not image_path.exists():
    print("Please add a traffic image named traffic.jpg")
else:
   results = model.predict(
    source=str(image_path),
    conf=0.10,
    save=True,
    show_labels=True,
    show_conf=True,
)

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        print(
            f"Class: {model.names[class_id]}, "
            f"Confidence: {confidence:.2f}, "
            f"Box: ({x1}, {y1}, {x2}, {y2})"
        )
