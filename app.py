from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference_sdk import InferenceHTTPClient
import shutil
import uuid
import os
import tempfile
app = FastAPI(title="Palm Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔴 Replace with your own Roboflow API key
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="W5Oc0aWD7MFYfm22hoIs"
)

MODEL_ID = "palm-lines-recognition-azgh0/5"


@app.get("/")
def health_check():
    return {"status": "ok"}




@app.post("/analyze-palm")
async def analyze_palm(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    temp_filename = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            temp_filename = tmp.name
            shutil.copyfileobj(file.file, tmp)

        result = CLIENT.infer(temp_filename, model_id=MODEL_ID)

        predictions = result.get("predictions", [])

        formatted = []
        for p in predictions:
            formatted.append({
                "label": p.get("class"),
                "confidence": round(p.get("confidence", 0), 2),
                "x": p.get("x"),
                "y": p.get("y"),
                "width": p.get("width"),
                "height": p.get("height")
            })

        return {"lines": formatted}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)
