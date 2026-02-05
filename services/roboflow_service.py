from inference_sdk import InferenceHTTPClient
import os

MODEL_ID = "palm-lines-recognition-azgh0/5"

def detect_palm_lines(image_path: str):
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY not set")

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
        timeout=10
    )

    result = client.infer(
        image_path=image_path,
        model_id=MODEL_ID
    )

    # Roboflow already trained hai → yahin se labels aate hain
    return result.get("predictions", [])
