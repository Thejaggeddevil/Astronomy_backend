import requests
import os

MODEL_ID = "palm-lines-recognition-azgh0/5"
API_URL = "https://serverless.roboflow.com"

def detect_palm_lines(image_path: str):
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY not set")

    url = f"{API_URL}/{MODEL_ID}?api_key={api_key}"

    with open(image_path, "rb") as f:
        files = {
            "file": f
        }

        response = requests.post(url, files=files)

    if response.status_code != 200:
        raise RuntimeError(
            f"Roboflow error {response.status_code}: {response.text}"
        )

    data = response.json()
    return data.get("predictions", [])
