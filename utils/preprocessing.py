"""
Preprocessing pipeline — matches the paper exactly:
1. Grayscale thresholding to find cell region
2. Center crop with square padding (removes black background)
3. CLAHE on Y channel of YUV colour space
4. Resize to 224x224 for model input
"""

import cv2
import numpy as np
from PIL import Image


def remove_black_background(img_rgb: np.ndarray) -> np.ndarray:
    """
    Converts to grayscale, thresholds to find non-black pixels,
    then crops to a square bounding box around the cell.
    Matches Section 3.1.2 of the paper.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        # Image is entirely black — return as-is
        return img_rgb

    x, y, w, h = cv2.boundingRect(coords)

    # Square padding — take the larger dimension as the side length
    side = max(w, h)
    cx = x + w // 2
    cy = y + h // 2

    x1 = max(cx - side // 2, 0)
    y1 = max(cy - side // 2, 0)
    x2 = min(x1 + side, img_rgb.shape[1])
    y2 = min(y1 + side, img_rgb.shape[0])

    cropped = img_rgb[y1:y2, x1:x2]
    return cropped


def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE only to the Y channel in YUV colour space.
    Matches Section 3.1.2 of the paper (Figure 4).
    """
    yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return enhanced


def preprocess_image(pil_image: Image.Image, target_size: int = 224) -> np.ndarray:
    """
    Full preprocessing pipeline from PIL Image → numpy array
    ready for model input.

    Returns:
        np.ndarray of shape (224, 224, 3), dtype float32, values in [0, 1]
    """
    img = np.array(pil_image.convert("RGB"))

    img = remove_black_background(img)
    img = apply_clahe(img)
    img = cv2.resize(img, (target_size, target_size))

    img = img.astype(np.float32) / 255.0
    return img


def get_intermediate_images(pil_image: Image.Image):
    """
    Returns each stage of preprocessing as PIL Images
    so Streamlit can display the pipeline visually.
    """
    original = np.array(pil_image.convert("RGB"))

    cropped = remove_black_background(original)
    enhanced = apply_clahe(cropped)
    resized = cv2.resize(enhanced, (224, 224))

    return (
        Image.fromarray(original),
        Image.fromarray(cropped),
        Image.fromarray(enhanced),
        Image.fromarray(resized),
    )
