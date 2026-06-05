# src/preprocessing.py
"""
Stage 1 of the absence pipeline — image cleaning.

Responsibilities:
    1. Load the raw image from disk.
    2. Extract the A4 page from its background (border removal).
    3. Convert to grayscale (modular, reused internally).
    4. Correct rotation (deskew).
    5. Improve local contrast (CLAHE).
    6. Binarize (adaptive threshold).
    7. Save the cleaned image to data/preprocessed/.

This module does NOT split the image into header and table regions.
That is the responsibility of cropper.py (Stage 2).

Input  : data/raw/real/as1.jpeg  (raw photograph of one page)
Output : data/preprocessed/as1.jpeg  (cleaned page, saved to disk)
         + returns a dict of intermediate arrays for the next stage.
"""

import os
import cv2
import numpy as np
from typing import Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(image_path: str,
               output_dir: str = "data/preprocessed") -> Dict[str, object]:
    """
    Full cleaning pipeline for one page of an attendance sheet.

    Args:
        image_path : path to the raw scanned/photographed page.
        output_dir : folder where the cleaned image will be saved.
                     Created automatically if it does not exist.

    Returns:
        {
          'original'    : (h,  w,  3) uint8  raw loaded image, untouched
          'cleaned'     : (h', w', 3) uint8  cleaned BGR image
                          (may be smaller than original after page extraction)
          'binary'      : (h', w')    uint8  binary version of cleaned image
          'output_path' : str         path where the cleaned image was saved
        }

    Raises:
        FileNotFoundError if image_path does not point to a readable file.
    """
    image    = _load(image_path)
    page     = _extract_page(image)
    gray     = _to_grayscale(page)
    deskewed = _deskew(page, gray)
    enhanced = _enhance(deskewed)
    binary   = _binarize(enhanced)

    os.makedirs(output_dir, exist_ok=True)
    filename    = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, enhanced)

    return {
        'original':    image,
        'cleaned':     enhanced,
        'binary':      binary,
        'output_path': output_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: str) -> np.ndarray:
    """
    Read an image file from disk into a NumPy array.

    IN  : file path string
    OUT : (h, w, 3) uint8  BGR color array

    cv2.imread decodes the file and returns a NumPy array in BGR order.
    It returns None silently on any failure (wrong path, unsupported
    format, permission error) — it never raises on its own.
    We detect None and raise a meaningful error.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(
            f"Could not load image. Verify the path exists and is readable:\n"
            f"  {path}"
        )
    return image


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR color image to a single-channel grayscale image.

    IN  : (h, w, 3) uint8  BGR
    OUT : (h, w)    uint8  grayscale

    cv2.cvtColor applies the weighted formula:
        gray = 0.114 × B  +  0.587 × G  +  0.299 × R
    The weights reflect human visual perception (most sensitive to green).
    Each 3-value pixel collapses to one integer 0–255 (luminance).

    This is a standalone function — not inlined — because grayscale
    conversion is needed in multiple places (_extract_page, _deskew,
    _binarize). Keeping it separate avoids duplication and makes
    each step independently testable.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _extract_page(image: np.ndarray,
                  margin: int = 0,
                  inner_crop_ratio: float = 0.02) -> np.ndarray:
    """
    Detect and crop to just the white A4 page, removing the background
    AND the thin inner border of the page itself.

    IN  : (h, w, 3) uint8  raw photograph (page + surrounding background)
          inner_crop_ratio fraction of page dimensions to trim from each side
    OUT : (h', w', 3) uint8  cropped to page core content only

    ── WHY TWO‑STAGE CROPPING ───────────────────────────────────────────
    The input images are photographs. We need to:
        1. Remove the desk/background outside the A4 sheet.
        2. Remove the thin white margin inside the page edge.
    Stage 1 uses contour‑based detection (unchanged from the previous
    version). Stage 2 trims a small percentage from each side.
    """
    gray    = _to_grayscale(image)
    h, w    = gray.shape

    # ── Stage 1: desk / background removal ────────────────────────────────
    # Heavy blur turns text into grey, leaving the whole page as a bright blob.
    blurred = cv2.GaussianBlur(gray, (51, 51), 0)

    # Threshold at 150 (slightly lower to catch shadowed pages).
    _, page_mask = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    # Fill holes inside the page with a large closing operation.
    # Kernel size 100×100 works better for photos with strong shadows.
    fill_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
    filled = cv2.morphologyEx(page_mask, cv2.MORPH_CLOSE, fill_kernel)

    contours, _ = cv2.findContours(
        filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image

    page_contour = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(page_contour)

    if bw < w * 0.3 or bh < h * 0.3:
        return image

    x1 = int(np.clip(x - margin, 0, w))
    y1 = int(np.clip(y - margin, 0, h))
    x2 = int(np.clip(x + bw + margin, 0, w))
    y2 = int(np.clip(y + bh + margin, 0, h))
    cropped_outer = image[y1:y2, x1:x2]

    # ── Stage 2: inner page‑edge trim ──────────────────────────────────────
    # Remove `inner_crop_ratio` % from each side to eliminate the white
    # margin between the paper edge and the printed content.
    ch, cw = cropped_outer.shape[:2]
    dx = int(cw * inner_crop_ratio)
    dy = int(ch * inner_crop_ratio)

    return cropped_outer[dy:ch-dy, dx:cw-dx]


def _deskew(image: np.ndarray,
            gray: np.ndarray,
            max_skew_angle: float = 10.0) -> np.ndarray:
    """
    Detect and correct the rotation angle of the page.

    IN  : image (h, w, 3) uint8  color image to rotate
          gray  (h, w)    uint8  grayscale version (passed in to avoid
                                 recomputing what the caller already has)
          max_skew_angle         angles above this are ignored to prevent
                                 overcorrection on pages with mostly vertical
                                 content
    OUT : (h, w, 3) uint8  same shape, rotation corrected

    ── WHY HOUGH LINES ──────────────────────────────────────────────────
    Hough Line Transform asks "what angle are the detected line segments?"
    We filter for near-horizontal lines (the printed table rows) and take
    their median angle. This is the correct question for a document.
    """
    _, binary_inv = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    merged   = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, h_kernel)

    lines = cv2.HoughLinesP(
        merged, 1, np.pi / 180,
        threshold=150,
        minLineLength=80,
        maxLineGap=15
    )

    if lines is None:
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < max_skew_angle:
            angles.append(angle)

    if not angles:
        return image

    skew_angle = float(np.median(angles))
    h, w   = image.shape[:2]
    center = (w // 2, h // 2)
    M      = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def _enhance(image: np.ndarray,
             clip_limit: float = 2.0,
             tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Improve local contrast using CLAHE on the L‑channel of LAB.
    Leaves colors untouched.
    """
    lab            = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b        = cv2.split(lab)
    clahe          = cv2.createCLAHE(clipLimit=clip_limit,
                                     tileGridSize=tile_size)
    l_enhanced     = clahe.apply(l)
    lab_enhanced   = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def _binarize(image: np.ndarray,
              block_size: int = 31,
              C: int = 15) -> np.ndarray:
    """
    Convert to pure black/white using adaptive Gaussian thresholding.
    Returns text=0 (black), background=255 (white).
    """
    gray    = _to_grayscale(image) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary_inv = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, C
    )
    return cv2.bitwise_not(binary_inv)


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND‑LINE TEST (all debug output goes to data/preprocessed/)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/preprocessing.py <image_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    result = preprocess(input_path)

    out_dir = "data/preprocessed"
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]

    # The cleaned image is already saved by preprocess().
    # Optionally save intermediate images for inspection.
    cv2.imwrite(os.path.join(out_dir, f"{base}_original.jpg"), result['original'])
    cv2.imwrite(os.path.join(out_dir, f"{base}_binary.jpg"),   result['binary'])

    print(f"Cleaned image → {result['output_path']}")
    print(f"Original shape : {result['original'].shape}")
    print(f"Cleaned shape  : {result['cleaned'].shape}")
    print("Debug images saved in", out_dir)