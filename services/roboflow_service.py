import requests

API_KEY = "W5Oc0aWD7MFYfm22hoIs"
MODEL_ID = "palm-lines-recognition-azgh0/5"

def detect_palm_lines(image_path: str):
    url = f"https://detect.roboflow.com/{MODEL_ID}"
    params = {
        "api_key": API_KEY
    }

    with open(image_path, "rb") as f:
        response = requests.post(
            url,
            params=params,
            files={"file": f},
            timeout=30
        )

    response.raise_for_status()
    data = response.json()

    return data.get("predictions", [])
