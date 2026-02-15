from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="W5Oc0aWD7MFYfm22hoIs"

)

result = client.infer("palm.jpg", model_id="palm-lines-recognition-azgh0/5")

print(result)
