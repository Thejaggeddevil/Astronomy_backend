from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI()

# -------- LOAD MODEL ONCE --------
MODEL_PATH = "palmdetector.pt"  # use ONE, not both
device = "cpu"

model = torch.load(MODEL_PATH, map_location=device)
model.eval()

LABEL_MAP = {
    0: "life",
    1: "head",
    2: "heart"
}

# -------- HELPERS --------
def preprocess(img: np.ndarray):
    img = cv2.resize(img, (640, 640))
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img

def mask_center(mask: np.ndarray):
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())

# -------- API --------
@app.post("/analyze-palm")
async def analyze_palm(image: UploadFile = File(...)):
    data = await image.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = np.array(img)

    inp = preprocess(img)

    with torch.no_grad():
        output = model(inp)

    results = []

    # 🔴 THIS PART DEPENDS ON MODEL TYPE
    # assuming segmentation-style output
    masks = output["masks"] if isinstance(output, dict) else output

    h, w = img.shape[:2]

    for idx, mask in enumerate(masks[:3]):  # only top 3
        mask = mask.squeeze().cpu().numpy()
        center = mask_center(mask)
        if not center:
            continue

        cx, cy = center
        results.append({
            "label": LABEL_MAP.get(idx, "unknown"),
            "confidence": float(mask.max()),
            "x": cx / mask.shape[1],
            "y": cy / mask.shape[0]
        })

    return JSONResponse({"lines": results})
