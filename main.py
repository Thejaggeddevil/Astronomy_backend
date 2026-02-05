from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os, uuid, shutil

from services.roboflow_service import detect_palm_lines

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join(TEMP_DIR, filename)

    with open(path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    try:
        predictions = detect_palm_lines(path)

        lines = []
        for p in predictions:
            lines.append({
                "label": p.get("class"),
                "confidence": round(p.get("confidence", 0), 2),
                "x": p.get("x"),
                "y": p.get("y")
            })

        return JSONResponse({"lines": lines})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(path):
            os.remove(path)
