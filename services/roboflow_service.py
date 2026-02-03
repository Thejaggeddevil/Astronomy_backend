from inference_sdk import InferenceHTTPClient
import os

MODEL_ID = "palm-lines-recognition-azgh0/5"

def detect_palm_lines(image_path: str):
    try:
        api_key = os.getenv("ROBOFLOW_API_KEY")

        if not api_key:
            print("❌ ROBOFLOW_API_KEY NOT SET")
            return []

        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
            timeout=15
        )

        result = client.infer(
            image_path=image_path,
            model_id=MODEL_ID
        )

        predictions = result.get("predictions", [])
        print(f"✅ Roboflow predictions count: {len(predictions)}")

        return predictions

    except Exception as e:
        print("❌ Roboflow error:", e)
        return []
