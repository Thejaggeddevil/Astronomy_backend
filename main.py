from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid

from services.roboflow_service import detect_palm_lines
from services.analysis_service import generate_analysis

app = FastAPI(title="Palmistry Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/analyze-palm")
async def analyze_palm(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    filename = f"{uuid.uuid4()}_{image.filename}"
    image_path = os.path.join(TEMP_DIR, filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        predictions = detect_palm_lines(image_path)
        analysis = generate_analysis(predictions)
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

    return {
        "predictions": predictions,
        "analysis": analysis
    }
