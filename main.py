from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, shutil
import cv2


from services.palm_cv_service import detect_palm_lines_cv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/analyze-palm")
async def analyze_palm(image: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(TEMP_DIR, filename)

    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    output_img = detect_palm_lines_cv(image_path)

    out_path = image_path.replace(".jpg", "_out.jpg")
    cv2.imwrite(out_path, output_img)

    return FileResponse(
        out_path,
        media_type="image/jpeg"
    )
