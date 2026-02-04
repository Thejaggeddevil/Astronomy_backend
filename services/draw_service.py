import cv2

LABELS = {
    "life": "Life Line",
    "heart": "Heart Line",
    "head": "Head Line",
    "fate": "Fate Line"
}

def draw_predictions(image_path: str, predictions: list):
    img = cv2.imread(image_path)

    if img is None:
        raise RuntimeError("Failed to load image")

    h, w, _ = img.shape

    for p in predictions:
        cls = p["class"]
        x = int(p["x"])
        y = int(p["y"])

        label = LABELS.get(cls, cls)

        # draw circle
        cv2.circle(img, (x, y), 8, (0, 255, 0), -1)

        # draw label
        cv2.putText(
            img,
            label,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    output_path = image_path.replace(".jpg", "_out.jpg")
    cv2.imwrite(output_path, img)

    return output_path
