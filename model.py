import torch
from PIL import Image
import os

from class_labels import CLASS_LABELS

# Load model ONCE (important for Render)
MODEL_PATH = "palmdetector.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(MODEL_PATH, map_location=device)
model.eval()


def analyze_image(image_path: str):
    """
    Returns a LIST OF POINTS.
    Each point = one detection center.
    """

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Run inference (YOLO-style assumption)
    results = model(image)

    detections = []

    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls_id = det.tolist()

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        detections.append({
            "label": CLASS_LABELS.get(int(cls_id), "unknown"),
            "confidence": round(conf, 2),
            # NORMALIZED coordinates (IMPORTANT)
            "x": cx / width,
            "y": cy / height
        })

    return detections
