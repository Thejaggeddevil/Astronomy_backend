import cv2
import numpy as np

import cv2
import numpy as np

def resize_keep_aspect(img, max_size=900):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))

def detect_palm_lines_cv(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Image could not be loaded")

    img = resize_keep_aspect(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    edges = cv2.Canny(thresh, 50, 150)
    edges = cv2.medianBlur(edges, 5)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    drawn = img.copy()

    for cnt in contours:
        length = cv2.arcLength(cnt, False)
        if length < 140:
            continue

        cv2.drawContours(drawn, [cnt], -1, (0, 255, 0), 2)

    return drawn
