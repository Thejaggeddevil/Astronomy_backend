from ultralytics import YOLO
from PIL import Image

from class_labels import CLASS_LABELS

MODEL_PATH = "palmdetector.pt"

# Load model ONCE at startup
model = YOLO(MODEL_PATH)


def analyze_image(image_path: str):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    results = model(image)[0]

    detections = []

    if results.boxes is None:
        return detections

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        detections.append({
            "label": CLASS_LABELS.get(cls_id, "unknown"),
            "confidence": round(conf, 2),
            "x": cx / width,
            "y": cy / height
        })

    return detections
