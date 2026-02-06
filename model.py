import torch
from PIL import Image
import os

from class_labels import CLASS_LABELS

MODEL_PATH = "palmdetector.pth"   # use .pth if that’s the state_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# LOAD MODEL CORRECTLY
# -------------------------------

def load_model():
    """
    Loads a YOLO-style model from state_dict safely.
    """
    # Example: YOLOv5 via torch hub
    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=MODEL_PATH,
        force_reload=False
    )

    model.to(device)
    model.eval()
    return model


model = load_model()


# -------------------------------
# INFERENCE
# -------------------------------

def analyze_image(image_path: str):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Run inference
    results = model(image)

    detections = []

    # YOLOv5 results format
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls_id = det.tolist()

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        detections.append({
            "label": CLASS_LABELS.get(int(cls_id), "unknown"),
            "confidence": round(conf, 2),
            "x": cx / width,   # normalized
            "y": cy / height
        })

    return detections
