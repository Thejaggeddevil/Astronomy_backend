def generate_analysis(predictions):
    if not predictions:
        return "No palm lines detected clearly."

    line_strengths = []

    for p in predictions:
        cls = p.get("class")
        conf = p.get("confidence", 0)

        if cls not in ["life", "heart", "head", "fate"]:
            continue

        if conf >= 0.7:
            strength = "Strong"
        elif conf >= 0.4:
            strength = "Moderate"
        else:
            strength = "Weak"

        line_strengths.append(f"{cls.capitalize()} Line: {strength}")

    if not line_strengths:
        return "Palm lines not clear enough for analysis."

    summary = ", ".join(line_strengths)

    return (
        "Palm Analysis Summary:\n"
        f"{summary}\n\n"
        "Future Guidance:\n"
        "- Strong lines indicate stability and confidence in life decisions.\n"
        "- Moderate lines suggest adaptability and gradual growth.\n"
        "- Weak lines indicate areas where extra effort and awareness are needed.\n\n"
        "This analysis reflects life tendencies, not fixed outcomes."
    )
