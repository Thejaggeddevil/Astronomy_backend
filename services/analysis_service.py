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
