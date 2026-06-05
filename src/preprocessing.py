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
                  margin: int = 6) -> np.ndarray:
    """
    Detect and crop to just the white A4 page, removing the background.

    IN  : (h, w, 3) uint8  raw photograph (page + surrounding background)
          margin            pixels of padding kept around the detected page
    OUT : (h', w', 3) uint8  cropped to the page only

    ── WHY THIS IS NEEDED ───────────────────────────────────────────────
    The input images are photographs taken with a phone or scanner.
    The white A4 page sits on a surface (desk, table) that is distinctly
    darker than the white paper. Everything outside the page boundary is
    noise that would corrupt all downstream processing.

    ── APPROACH ─────────────────────────────────────────────────────────
    1. Convert to grayscale.
    2. Blur heavily — we want to see the page shape, not its content.
       A large blur kernel turns text (black pixels) into grey, so the
       entire page region becomes one uniformly bright blob.
    3. Threshold at a moderate fixed value — the bright page blob
       separates from the darker background.
    4. Morphological close with a large kernel — fills the remaining
       dark patches (text, table lines) inside the page region, making
       the page one solid white contour.
    5. Find contours — the page should be the largest contour by area.
    6. Get its bounding rectangle — (x, y, width, height).
    7. Crop the ORIGINAL color image to that rectangle, plus a small margin
       to avoid accidentally cutting edge content.

    ── FALLBACK ─────────────────────────────────────────────────────────
    If no suitable contour is found, the original image is returned
    unchanged. This is safe — subsequent steps will still function,
    just with the extra border included.
    """
    gray    = _to_grayscale(image)
    h, w    = gray.shape

    # ── Step 2: heavy blur ──────────────────────────────────────────────
    # Kernel (51, 51): large enough to blur individual characters into grey.
    # The integer 51 must be odd. Larger = more blur = smoother page shape.
    # 0: standard deviation computed automatically from kernel size.
    # IN : (h, w) grayscale with text detail
    # OUT: (h, w) grayscale, text blurred away, page appears as bright blob
    blurred = cv2.GaussianBlur(gray, (51, 51), 0)

    # ── Step 3: fixed threshold to separate page from background ────────
    # THRESH_BINARY: pixels >= 180 → 255 (white), below → 0 (black)
    # 180 is chosen as a moderate value:
    #   - The white page, even after blurring, stays near 200–250
    #   - The darker background (desk, table) typically falls below 180
    # We use a fixed value rather than Otsu here because Otsu would find
    # the threshold between text and background within the page, which is
    # not what we want.
    # IN : (h, w) blurred grayscale
    # OUT: (h, w) binary — white = page region, black = background
    _, page_mask = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

    # ── Step 4: morphological close to fill holes (text, lines) ─────────
    # Text inside the page appears as dark pixels, creating holes in the
    # white page mask. We fill them by closing with a large kernel (80×80).
    # MORPH_CLOSE = dilation then erosion:
    #   dilation expands white regions, merging holes
    #   erosion restores the outer boundary
    # Result: the entire page interior becomes one solid white region.
    fill_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
    filled = cv2.morphologyEx(page_mask, cv2.MORPH_CLOSE, fill_kernel)

    # ── Step 5 & 6: find largest contour and its bounding rectangle ──────
    # cv2.findContours scans the binary image for connected white regions.
    # RETR_EXTERNAL: return only outermost contours (no nested ones).
    # CHAIN_APPROX_SIMPLE: compress segments to only their endpoints.
    # Returns: (contours_list, hierarchy)
    contours, _ = cv2.findContours(
        filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        # No contour found — return image as-is
        return image

    # cv2.contourArea returns the area in pixels² for each contour.
    # max(..., key=cv2.contourArea) picks the contour with the largest area.
    # For a photograph of an A4 page, the page itself is by far the largest
    # white connected region.
    page_contour = max(contours, key=cv2.contourArea)

    # cv2.boundingRect returns (x, y, width, height) of the upright
    # bounding rectangle around the contour.
    x, y, bw, bh = cv2.boundingRect(page_contour)

    # Sanity check: the detected region must be at least 30% of the image
    # in both dimensions. If not, the detection failed — return as-is.
    if bw < w * 0.3 or bh < h * 0.3:
        return image

    # ── Step 7: crop to detected page, plus margin ───────────────────────
    # np.clip ensures coordinates stay within the image bounds.
    # np.clip(value, min, max) — clamps value between min and max.
    x1 = int(np.clip(x - margin, 0, w))
    y1 = int(np.clip(y - margin, 0, h))
    x2 = int(np.clip(x + bw + margin, 0, w))
    y2 = int(np.clip(y + bh + margin, 0, h))

    # Slice the original BGR image to the detected page region.
    # image[y1:y2, x1:x2] — rows y1 to y2, columns x1 to x2
    return image[y1:y2, x1:x2]


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
    A previous version used cv2.minAreaRect on all dark pixels.
    That caused 90° overcorrections because it asked "what angle is this
    entire blob of dark pixels?" — which is dominated by vertical content.

    Hough Line Transform asks "what angle are the detected line segments?"
    We filter for near-horizontal lines (the printed table rows) and take
    their median angle. This is always the correct question for a document.

    ── STEPS ────────────────────────────────────────────────────────────
    1. Invert-binarize the grayscale image.
       Hough needs white objects on black — so text becomes white.
    2. Morphological close with a horizontal kernel.
       This merges individual characters on the same text line into one
       solid horizontal bar, making Hough detection reliable.
    3. HoughLinesP — detect line segments in the merged image.
    4. Filter — keep only near-horizontal segments.
    5. Median angle of kept segments = estimated page skew.
    6. warpAffine — rotate the original color image to correct the skew.
    """

    # ── Step 1: inverted binary ──────────────────────────────────────────
    # THRESH_BINARY_INV: text → 255 (white), background → 0 (black)
    # THRESH_OTSU: auto-finds the best global threshold
    # _: we discard the computed threshold value, keeping only the image
    _, binary_inv = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ── Step 2: merge characters into solid lines ────────────────────────
    # Kernel (40, 1): 40 pixels wide, 1 pixel tall = horizontal ruler.
    # MORPH_CLOSE = dilation then erosion:
    #   dilation expands white pixels 40px horizontally → gaps filled
    #   erosion shrinks back → characters on same line merge into one bar
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    merged   = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, h_kernel)

    # ── Step 3: Hough Line Transform ────────────────────────────────────
    # Returns array of shape (N, 1, 4): N line segments.
    # Each segment: [[x1, y1, x2, y2]] — pixel coordinates of two endpoints.
    # Returns None if no segments found.
    #
    # Parameters:
    #   1              distance resolution: 1-pixel precision
    #   np.pi / 180    angle resolution: test every 1°
    #                  (np.pi radians = 180°, so /180 = 1° in radians)
    #   threshold=150  minimum pixel votes to accept a line
    #   minLineLength=80  discard segments shorter than 80px
    #   maxLineGap=15     bridge gaps under 15px within one segment
    lines = cv2.HoughLinesP(
        merged, 1, np.pi / 180,
        threshold=150,
        minLineLength=80,
        maxLineGap=15
    )

    if lines is None:
        return image

    # ── Steps 4 & 5: filter and compute median angle ─────────────────────
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # np.arctan2(dy, dx) → angle in radians; np.degrees() → degrees
        # Horizontal line = 0°; tilted 3° clockwise = −3°
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < max_skew_angle:
            angles.append(angle)

    if not angles:
        return image

    # np.median: unaffected by outlier lines — takes the middle value
    skew_angle = float(np.median(angles))

    # ── Step 6: rotate ───────────────────────────────────────────────────
    h, w   = image.shape[:2]
    center = (w // 2, h // 2)   # rotation pivot = image center
    M      = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    # getRotationMatrix2D: builds a 2×3 affine matrix for rotation
    # warpAffine: applies M to every pixel
    # INTER_CUBIC: bicubic interpolation (smooth result)
    # BORDER_REPLICATE: fills corners by repeating nearest border pixel
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def _enhance(image: np.ndarray,
             clip_limit: float = 2.0,
             tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Improve local contrast to make faint handwriting and lines clearer.

    IN  : (h, w, 3) uint8  BGR — may have uneven lighting or faint ink
    OUT : (h, w, 3) uint8  BGR — same shape, improved local contrast

    ── WHY LAB COLOR SPACE ──────────────────────────────────────────────
    In BGR, brightness and color are entangled across all channels.
    Adjusting brightness changes hue and saturation as a side effect.
    LAB separates them: L = lightness only, A and B = color axes.
    We enhance only L → colors preserved, brightness improved.

    ── WHY CLAHE ────────────────────────────────────────────────────────
    equalizeHist applies one global correction to the entire image.
    CLAHE (Contrast Limited Adaptive Histogram Equalization) divides
    the image into tiles and equalizes each independently.
    This handles shadows and lighting gradients correctly.
    clip_limit caps amplification per tile to avoid noise explosion.

    ── STEPS ────────────────────────────────────────────────────────────
    BGR → LAB → split → apply CLAHE to L only → merge → LAB → BGR
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
    Convert an image to pure black/white (binary).

    IN  : BGR image (h, w, 3)  OR  grayscale (h, w)
    OUT : (h, w) uint8  — text = 0 (black), background = 255 (white)

    ── WHY ADAPTIVE THRESHOLD ───────────────────────────────────────────
    Otsu computes ONE global threshold for the entire image.
    With uneven lighting, the optimal threshold varies across regions.
    Adaptive Gaussian threshold computes a different threshold per pixel
    based on a local neighborhood — handles lighting variation correctly.

    block_size = 31  size of the local neighborhood (must be odd)
    C = 15           constant subtracted from local mean.
                     Higher C → sparser result (fewer black pixels).

    ── STEPS ────────────────────────────────────────────────────────────
    1. Grayscale (if input is color)
    2. GaussianBlur — slight smoothing to reduce noise impact
    3. adaptiveThreshold (inverted: text → white)
    4. bitwise_not — invert to standard convention (text → black)
    """
    gray    = _to_grayscale(image) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # adaptiveThreshold:
    # ADAPTIVE_THRESH_GAUSSIAN_C — local threshold = Gaussian-weighted
    #                              neighborhood mean minus C
    # THRESH_BINARY_INV — below threshold → 255, above → 0
    # IN : (h, w) grayscale  OUT: (h, w) text=255, bg=0
    binary_inv = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, C
    )

    # bitwise_not: 0 ↔ 255 — invert to text=0, bg=255
    return cv2.bitwise_not(binary_inv)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# Usage: python src/preprocessing.py data/raw/real/as1.jpeg
# Saves debug_*.jpg in the current directory for visual inspection.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/preprocessing.py <image_path>")
        sys.exit(1)

    result = preprocess(sys.argv[1])
    print(f"Saved cleaned image → {result['output_path']}")
    print(f"Original shape : {result['original'].shape}")
    print(f"Cleaned shape  : {result['cleaned'].shape}")

    cv2.imwrite("debug_original.jpg", result['original'])
    cv2.imwrite("debug_cleaned.jpg",  result['cleaned'])
    cv2.imwrite("debug_binary.jpg",   result['binary'])
    print("Saved: debug_original.jpg  debug_cleaned.jpg  debug_binary.jpg")