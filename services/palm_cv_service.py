import cv2
import numpy as np

import cv2
import numpy as np

def resize_keep_aspect(img, max_size=900):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))

dimport cv2
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

    # 1️⃣ Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 2️⃣ Blur skin noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3️⃣ Edge detection (lighter thresholds)
    edges = cv2.Canny(blurred, 30, 90)

    # 4️⃣ Morphological thinning (connect broken lines)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5️⃣ Skeletonize (THIS IS THE KEY STEP)
    skel = np.zeros(edges.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        open_ = cv2.morphologyEx(edges, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(edges, open_)
        eroded = cv2.erode(edges, element)
        skel = cv2.bitwise_or(skel, temp)
        edges = eroded.copy()
        if cv2.countNonZero(edges) == 0:
            break

    # 6️⃣ Find contours on skeleton
    contours, _ = cv2.findContours(
        skel,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    drawn = img.copy()

    for cnt in contours:
        length = cv2.arcLength(cnt, False)

        # MUCH LOWER threshold (palm lines are thin)
        if length < 60:
            continue

        cv2.drawContours(drawn, [cnt], -1, (0, 255, 0), 2)

    return drawn
