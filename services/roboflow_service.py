from PIL import Image
from inference_sdk import InferenceHTTPClient
import os

API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = "palm-lines-recognition-azgh0/5"

def detect_palm_lines(image_path: str):
    try:
        img = Image.open(image_path)
        img_w, img_h = img.size

        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=API_KEY,
            timeout=8
        )

        result = client.infer(image_path=image_path, model_id=MODEL_ID)

        predictions = []
        for p in result.get("predictions", []):
            predictions.append({
                "class": p["class"],
                "confidence": p["confidence"],
                # 🔥 NORMALIZED VALUES
                "x": p["x"] / img_w,
                "y": p["y"] / img_h,
                "radius": max(p["width"], p["height"]) / max(img_w, img_h)
            })

        return predictions

    except Exception as e:
        print("Roboflow error:", e)
        return []
