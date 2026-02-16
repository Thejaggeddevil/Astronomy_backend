from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from inference_sdk import InferenceHTTPClient
from groq import Groq
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

# ------------------ ROBOFLOW CONFIG ------------------

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")  # set in Render
)

MODEL_ID = "palm-lines-recognition-azgh0/5"

# ------------------ GROQ CONFIG ------------------

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")  # set in Render
)

# -----------------------------------------------------


@app.get("/")
def health_check():
    return {"status": "ok"}


# ===========================
# PALM DETECTION ENDPOINT
# ===========================

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
                "height": p.get("height"),
                "keypoints": p.get("keypoints", [])
            })

        return {"lines": formatted}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)


# ===========================
# FUTURE PREDICTION ENDPOINT
# ===========================

@app.post("/future-prediction")
async def future_prediction(
    date_of_birth: str = Form(...),
    birth_place: str = Form(...)
):
    try:

        prompt = f"""
        You are a professional life advisor.

        Based on:
        Date of Birth: {date_of_birth}
        Birth Place: {birth_place}

        Generate a structured response in JSON format with these fields:
        - personality
        - career
        - love
        - strengths
        - next_three_years

        Keep tone realistic, practical and motivating.
        Do not mention astrology explicitly.
        """

        response = groq_client.chat.completions.create(
           model="llama-3.1-8b-instant",

            messages=[
                {"role": "system", "content": "You are an intelligent life prediction assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        content = response.choices[0].message.content

        return {
            "prediction": content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
