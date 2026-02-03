def detect_palm_lines(image_path: str):
    if not API_KEY:
        print("❌ ROBOFLOW_API_KEY NOT SET")
        return []

    try:
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=API_KEY,
            timeout=8
        )

        result = client.infer(
            image_path=image_path,
            model_id=MODEL_ID
        )

        return result.get("predictions", [])

    except Exception as e:
        print("❌ Roboflow error:", e)
        return []
