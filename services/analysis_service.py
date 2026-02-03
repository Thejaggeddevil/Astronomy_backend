def generate_analysis(predictions):
    if not predictions:
        return "No palm lines detected clearly."

    line_strengths = []

    for p in predictions:
        cls = p.get("class")
        conf = p.get("confidence", 0)

        if cls not in ["life", "heart", "head", "fate"]:
            continue

        # 🔥 FIXED THRESHOLDS
        if conf >= 0.45:
            strength = "Strong"
        elif conf >= 0.25:
            strength = "Moderate"
        else:
            strength = "Weak"

        line_strengths.append({
            "name": cls,
            "strength": strength,
            "x": p["x"],   # DO NOT CHANGE
            "y": p["y"]    # DO NOT CHANGE
        })

    if not line_strengths:
        return "Palm lines not clear enough for analysis."

    return {
        "lines": line_strengths
    }
