from inference_sdk import InferenceHTTPClient
import os
import traceback

MODEL_ID = "palm-lines-recognition-azgh0/5"

def detect_palm_lines(image_path: str):
    api_key = os.getenv("ROBOFLOW_API_KEY")

    # ✅ SAFE GUARD (NO HARD CRASH)
    if not api_key:
        print("❌ ERROR: ROBOFLOW_API_KEY not found in environment")
        print("Available env keys:", list(os.environ.keys()))
        return []

    try:
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
            timeout=10
        )

        result = client.infer(
            image_path=image_path,
            model_id=MODEL_ID
        )

        predictions = result.get("predictions", [])

        print(f"✅ Roboflow inference success. Predictions: {len(predictions)}")

        return predictions

    except Exception as e:
        print("❌ Roboflow inference failed")
        print(str(e))
        traceback.print_exc()
        return []
