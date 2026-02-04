def analyze_lines(predictions: list):
    if not predictions:
        return {
            "message": "Palm lines not detected clearly",
            "lines": []
        }

    summary = []
    score = 0

    for p in predictions:
        line = p.get("class")
        confidence = p.get("confidence", 0)

        if line not in ["life", "heart", "head", "fate"]:
            continue

        if confidence >= 0.5:
            strength = "strong"
            score += 2
        elif confidence >= 0.3:
            strength = "moderate"
            score += 1
        else:
            strength = "weak"

        summary.append({
            "line": line,
            "strength": strength,
            "confidence": round(confidence, 2),
            "x": p["x"],
            "y": p["y"]
        })

    outlook = (
        "Very positive signs" if score >= 6 else
        "Balanced life indications" if score >= 3 else
        "Unclear or developing traits"
    )

    return {
        "outlook": outlook,
        "lines": summary
    }
