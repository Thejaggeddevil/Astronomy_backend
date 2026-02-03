from inference_sdk import InferenceHTTPClient
import os

API_KEY = os.getenv("ROBOFLOW_API_KEY")  # safer for deploy
MODEL_ID = "palm-lines-recognition-azgh0/5"

def detect_palm_lines(image_path: str):
    try:
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=API_KEY,
            timeout=8  # ⬅️ HARD TIMEOUT (IMPORTANT)
        )

        result = client.infer(
            image_path=image_path,
            model_id=MODEL_ID
        )

        return result.get("predictions", [])

    except Exception as e:
        print("❌ Roboflow error:", e)
        return []  # ⬅️ NEVER crash backend
