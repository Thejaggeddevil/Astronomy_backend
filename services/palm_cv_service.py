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
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 30, 90)

    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Skeletonization
    skel = np.zeros(edges.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        opened = cv2.morphologyEx(edges, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(edges, opened)
        eroded = cv2.erode(edges, element)
        skel = cv2.bitwise_or(skel, temp)
        edges = eroded.copy()
        if cv2.countNonZero(edges) == 0:
            break

    contours, _ = cv2.findContours(
        skel,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    output = img.copy()

    # Define palm-only vertical region (ignore fingers)
    palm_top = int(h * 0.28)
    palm_bottom = int(h * 0.78)

    for cnt in contours:
        length = cv2.arcLength(cnt, False)

        # 1️⃣ Keep only long lines
        if length < 180:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        cy = y + ch // 2

        # 2️⃣ Ignore finger area
        if cy < palm_top or cy > palm_bottom:
            continue

        # 3️⃣ Remove near-straight lines (borders, finger edges)
        approx = cv2.approxPolyDP(cnt, 0.02 * length, False)
        if len(approx) <= 3:
            continue

        # 4️⃣ Remove very vertical shapes
        if ch > cw * 3:
            continue

        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 3)

    return output
