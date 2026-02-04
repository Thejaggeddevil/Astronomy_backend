from inference_sdk import InferenceHTTPClient
import os

def detect_palm_lines(image_path: str):
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        print("❌ ROBOFLOW_API_KEY NOT SET")
        return []

    try:
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
            timeout=8
        )

        result = client.infer(
            image_path=image_path,
            model_id="palm-lines-recognition-azgh0/5"
        )

        return result.get("predictions", [])

    except Exception as e:
        print("❌ Roboflow error:", e)
        return []
