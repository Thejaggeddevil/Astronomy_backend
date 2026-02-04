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

    # 🔻 Reduce micro-noise (VERY IMPORTANT)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # 🔻 Softer contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 🔻 Stronger thresholds = fewer junk lines
    edges = cv2.Canny(blurred, 60, 160)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    output = img.copy()

    # 🔻 Palm-only region (ignore fingers + wrist)
    palm_top = int(h * 0.32)
    palm_bottom = int(h * 0.75)

    for cnt in contours:
        length = cv2.arcLength(cnt, False)

        # 1️⃣ Keep only very long lines
        if length < w * 0.35:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        cy = y + ch // 2

        # 2️⃣ Must lie inside palm region
        if cy < palm_top or cy > palm_bottom:
            continue

        # 3️⃣ Line must span horizontally
        if cw < w * 0.25:
            continue

        # 4️⃣ Reject near-vertical shapes
        if ch > cw * 1.2:
            continue

        # 5️⃣ Must be smooth (not zig-zag noise)
        approx = cv2.approxPolyDP(cnt, 0.01 * length, False)
        if len(approx) < 6:
            continue

        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 3)

    return output
