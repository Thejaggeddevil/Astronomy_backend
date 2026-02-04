from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
from fastapi.responses import FileResponse
from services.draw_service import draw_predictions

from services.roboflow_service import detect_palm_lines
from services.analysis_service import analyze_lines

app = FastAPI(title="Palmistry Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    filename = f"{uuid.uuid4()}_{image.filename}"
    image_path = os.path.join(TEMP_DIR, filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    predictions = detect_palm_lines(image_path)
    analysis = analyze_lines(predictions)

    output_image = draw_predictions(image_path, predictions)

    return FileResponse(
        output_image,
        media_type="image/jpeg",
        headers={
            "X-Analysis": str(analysis)
        }
    )
