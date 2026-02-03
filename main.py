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
    try:
        image_path = os.path.join(TEMP_DIR, image.filename)

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        predictions = detect_palm_lines(image_path)
        analysis = generate_analysis(predictions)

        return {
            "predictions": predictions,
            "analysis": analysis
        }

    except Exception as e:
        print("❌ BACKEND CRASH PREVENTED:", e)
        return {
            "predictions": [],
            "analysis": "Server error handled safely."
        }
