# src/preprocessing.py
"""
Preprocessing module for scanned attendance sheets.

═══════════════════════════════════════════════════════════════════════════════
DOCUMENT STRUCTURE (from real sheet inspection)
═══════════════════════════════════════════════════════════════════════════════

A complete attendance record spans 1 to N pages (4 in the provided dataset).

PAGE 1 ONLY — contains a metadata header above the student table:

    ┌────────────────────────────────────────────────────────────────────┐
    │  [logo]    Université Sultan Moulay Slimane               [logo]  │
    │          L'Ecole Supérieure de Technologie -- Fkih Ben Salah      │
    │       Liste de présence : Filière ... année - 2024-2025           │
    ├───────────┬───────────────────────────────────────────────────────┤
    │Enseignant │ <handwritten teacher name>                            │
    ├───────────┼──────────────────────┬───────────┬───────────────────┤
    │Module     │ <handwritten module> │ Elément   │ <handwritten>     │
    ├───────────┴──────────────────────┴─────┬─────┬─────┬─── ── ──── ┤
    │                                        │Séa1 │Séa2 │ ... │Séa10 │
    ├──────────────┬─────────────────────────┼─────┼─────┼─── ── ──── ┤
    │ N.B:         │ Date                    │04/06│     │            │
    │ A: Absence   │ Heure Début             │08h30│     │            │
    │ P: Présence  │ Heure Fin               │10h30│     │            │
    │              │ Type(Crs,TP,TD,Cnt,Exm) │ Crs │     │            │
    ├──────────────┼─────────────────────────┼─────┴─────┴─── ── ──── ┤
    │   N° Apo     │    Nom & Prénom         │Séa1 │Séa2 │ ... │Séa10 │
    ╠══════════════╪═════════════════════════╪═════╪═════╪═══ ══ ════ ╣
    │              │ ACHMAOUI JIHANE         │     │     │            │
    │              │ AIT ALI HANANE          │sign │     │            │  ← student rows
    │              │ ...                     │     │     │            │
    └──────────────┴─────────────────────────┴─────┴─────┴─── ── ──── ┘

PAGES 2, 3, 4 — student table only, no header:

    ┌──────────────┬─────────────────────────┬─────┬─────┬─── ── ──── ┐
    │              │ EL ABBASSI HIBA         │     │     │            │
    │              │ EL ABBASSI HAJAR        │sign │     │            │
    │              │ ...                     │     │     │            │
    └──────────────┴─────────────────────────┴─────┴─────┴─── ── ──── ┘

    NOTE: last row of the final page is labelled "EMARGEMENT" —
          it is not a student and must be skipped during detection.

═══════════════════════════════════════════════════════════════════════════════
PROCESSING PIPELINE (per page)
═══════════════════════════════════════════════════════════════════════════════

    raw image
        → _load          read file from disk into a BGR NumPy array
        → _to_grayscale  collapse 3 color channels to 1 intensity channel
        → _deskew        detect and correct rotation using Hough lines
        → _enhance       improve local contrast using CLAHE (LAB color space)
        → _binarize      convert to pure black/white using adaptive threshold
        → _detect_split  decide if this page has a header and where it ends
        → crop           produce header region (or None) and table region
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# Only this function is imported by other modules.
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(image_path: str) -> Dict[str, Optional[np.ndarray]]:
    """
    Full preprocessing pipeline for one page of an attendance sheet.

    Args:
        image_path: path to the scanned page image (JPG or PNG).

    Returns:
        Dictionary with the following keys:

        'original'      (h, w, 3) uint8
                        The raw image exactly as loaded from disk.
                        Kept for debugging and side-by-side comparison.

        'binary'        (h, w) uint8
                        Full-page binary image. Every pixel is either
                        0 (black = text/lines) or 255 (white = background).

        'has_header'    bool
                        True only for page 1 of a document.
                        False for all continuation pages (pages 2, 3, 4...).

        'header'        (h1, w, 3) uint8  OR  None
                        BGR crop of the metadata header region.
                        None when has_header is False.

        'header_binary' (h1, w) uint8  OR  None
                        Binary version of the header crop.
                        None when has_header is False.

        'table'         (h2, w, 3) uint8
                        BGR crop of the student table region.
                        For pages 2+ this is the entire image.

        'table_binary'  (h2, w) uint8
                        Binary version of the table crop.
    """
    # ── load ──────────────────────────────────────────────────────────
    image = _load(image_path)

    # ── grayscale (computed once, reused by deskew) ───────────────────
    gray = _to_grayscale(image)

    # ── deskew: correct rotation before any further processing ────────
    # We pass gray in so _deskew does not have to recompute it.
    deskewed = _deskew(image, gray)

    # ── enhance contrast on the deskewed color image ──────────────────
    enhanced = _enhance(deskewed)

    # ── binarize the full enhanced image ──────────────────────────────
    binary = _binarize(enhanced)

    # ── detect whether this page has a header and where it ends ───────
    has_header, split_y = _detect_split(binary)

    # ── crop into header and table regions ────────────────────────────
    if has_header:
        header        = enhanced[:split_y, :]
        table         = enhanced[split_y:, :]
        header_binary = _binarize(header)
        table_binary  = _binarize(table)
    else:
        header        = None
        header_binary = None
        table         = enhanced
        table_binary  = binary

    return {
        'original':      image,
        'binary':        binary,
        'has_header':    has_header,
        'header':        header,
        'header_binary': header_binary,
        'table':         table,
        'table_binary':  table_binary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# Prefixed with _ by convention: not intended to be imported by other modules.
# Each function does exactly one thing.
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: str) -> np.ndarray:
    """
    Read an image file from disk into a NumPy array.

    IN  : file path string
    OUT : (h, w, 3) uint8 array in BGR color order

    cv2.imread returns None silently on failure (wrong path, unsupported
    format, permission error) — it never raises an exception on its own.
    We check manually and raise a meaningful error.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(
            f"Could not load image. Check the path exists and is readable:\n"
            f"  {path}"
        )
    return image


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR color image to a single-channel grayscale image.

    IN  : (h, w, 3) uint8 BGR
    OUT : (h, w)    uint8 grayscale

    cv2.cvtColor applies the formula:
        gray = 0.114 * B  +  0.587 * G  +  0.299 * R
    The coefficients reflect human visual perception — we are more
    sensitive to green, less to blue. Each 3-value pixel becomes
    one integer 0–255 representing luminance.

    This function exists as a standalone helper because grayscale
    conversion is needed in multiple places (_deskew, _binarize).
    Isolating it avoids code duplication and makes each step testable.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _deskew(image: np.ndarray,
            gray: np.ndarray,
            max_skew_angle: float = 10.0) -> np.ndarray:
    """
    Detect and correct the rotation angle of a scanned page.

    IN  : image (h, w, 3) uint8 BGR  — the color image to rotate
          gray  (h, w)    uint8      — grayscale version (pre-computed)
          max_skew_angle             — angles larger than this are ignored
                                       (prevents overcorrection on pages
                                       with lots of vertical content)
    OUT : (h, w, 3) uint8 BGR — same shape, rotation corrected

    ── WHY HOUGH LINES, NOT minAreaRect ─────────────────────────────────
    An earlier version of this code used cv2.minAreaRect on all dark
    pixels to estimate rotation. That approach caused 90° overcorrections
    on portrait pages with more vertical content than horizontal.

    The Hough Line Transform directly detects line segments in the image.
    We specifically look for near-horizontal lines (the printed table rows)
    and compute their median angle. This is the correct question to ask
    for a document: "what angle are the text lines at?" — not "what angle
    is the bounding box of all dark pixels?"

    ── STEPS ────────────────────────────────────────────────────────────
    1. Binarize (inverted) — Hough needs white objects on black background.
    2. Morphological close  — merge individual characters into solid bars
                              so Hough can detect the text line as one line.
    3. HoughLinesP          — find line segments in the merged image.
    4. Filter               — keep only near-horizontal lines.
    5. Median angle         — robust estimate of the page skew.
    6. warpAffine           — rotate the original color image to correct it.
    """

    # ── Step 1: inverted binary ──────────────────────────────────────────
    # THRESH_BINARY_INV: text → white (255), background → black (0)
    # THRESH_OTSU: automatically finds the best threshold value
    # The _ discards the computed threshold value (we only need the image)
    _, binary_inv = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ── Step 2: merge characters on the same line into solid bars ────────
    # getStructuringElement creates a kernel — a small matrix defining
    # the shape used in morphological operations.
    # MORPH_RECT = rectangular kernel
    # (40, 1) = 40 pixels wide, 1 pixel tall = a horizontal ruler
    #
    # MORPH_CLOSE = dilation then erosion:
    #   dilation expands every white pixel 40px left and right
    #             → gaps between letters fill in
    #   erosion  shrinks everything back
    #             → result: one solid white bar per line of text
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    merged = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, h_kernel)

    # ── Step 3: Probabilistic Hough Line Transform ────────────────────────
    # Scans the binary image and returns line segments as endpoint pairs.
    #
    # Parameters:
    #   merged        — binary input (white objects on black)
    #   1             — distance resolution: 1-pixel precision
    #   np.pi / 180   — angle resolution: test every 1 degree
    #                   np.pi radians = 180 degrees, so /180 = 1 degree in radians
    #   threshold=150 — minimum pixel votes for a line to be accepted
    #   minLineLength=80  — discard segments shorter than 80px
    #   maxLineGap=15     — bridge gaps smaller than 15px within one segment
    #
    # Returns: array of shape (N, 1, 4) — N line segments
    #          each segment: [[x1, y1, x2, y2]] — two endpoint pixel coords
    #          Returns None if no segments found
    lines = cv2.HoughLinesP(
        merged, 1, np.pi / 180,
        threshold=150,
        minLineLength=80,
        maxLineGap=15
    )

    if lines is None:
        # No lines detected — image is likely already straight
        return image

    # ── Step 4: filter for near-horizontal lines and collect their angles ─
    angles = []
    for line in lines:
        # line has shape (1, 4) — [0] unwraps it to the 4 values directly
        x1, y1, x2, y2 = line[0]

        # np.arctan2(vertical_component, horizontal_component)
        # returns the angle in RADIANS of the vector from (x1,y1) to (x2,y2)
        # np.degrees() converts radians to degrees
        # horizontal line = 0°, tilted 3° clockwise = -3°
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # abs() gives the absolute value, so both +5° and -5° pass the check
        if abs(angle) < max_skew_angle:
            angles.append(angle)

    if not angles:
        # All detected lines were too steep — no meaningful skew to correct
        return image

    # ── Step 5: median angle ──────────────────────────────────────────────
    # np.median is more robust than np.mean here.
    # A few misdetected lines won't skew the result because median
    # is unaffected by outliers — it takes the middle value, not the average.
    skew_angle = float(np.median(angles))

    # ── Step 6: rotate the original color image ───────────────────────────
    h, w = image.shape[:2]
    # .shape returns (h, w, 3) for a color image
    # [:2] slices only the first two values — channel count not needed here
    # // is integer (floor) division — center must be whole pixel coordinates
    center = (w // 2, h // 2)

    # getRotationMatrix2D returns a 2×3 affine transformation matrix M
    # that describes: rotate skew_angle degrees around center, scale 1.0
    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)

    # warpAffine applies matrix M to every pixel in the image
    # (w, h) — output size matches input size
    # INTER_CUBIC — bicubic interpolation for smooth pixel resampling
    # BORDER_REPLICATE — fills corners created by rotation by repeating
    #                    the nearest border pixel (avoids black triangles)
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

    IN  : (h, w, 3) uint8 BGR — may have uneven lighting or faint ink
    OUT : (h, w, 3) uint8 BGR — same shape, improved local contrast

    ── WHY CLAHE, NOT equalizeHist ──────────────────────────────────────
    equalizeHist applies one correction to the entire image globally.
    A page with a shadow in one corner and bright light in another
    gets one threshold — it helps the bright area and hurts the dark one.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) divides
    the image into small tiles and equalizes each tile independently.
    clip_limit prevents over-amplification of noise in uniform regions.

    ── WHY LAB COLOR SPACE ──────────────────────────────────────────────
    In BGR, brightness and color are mixed in all three channels.
    Touching brightness inevitably changes hue and saturation.

    LAB separates them cleanly:
        L = lightness (brightness only)
        A = green-to-red axis
        B = blue-to-yellow axis

    We apply CLAHE only to L. A and B are untouched.
    Colors are preserved. Only brightness distribution changes.

    ── STEPS ────────────────────────────────────────────────────────────
    1. Convert BGR → LAB
    2. Split into 3 channels
    3. Apply CLAHE to L only
    4. Merge channels back
    5. Convert LAB → BGR
    """

    # ── Step 1: BGR → LAB ─────────────────────────────────────────────────
    # IN : (h, w, 3) BGR
    # OUT: (h, w, 3) LAB — same shape, different meaning per channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # ── Step 2: split into 3 separate single-channel arrays ───────────────
    # cv2.split takes a multi-channel array and returns a list of
    # single-channel arrays, one per channel
    # IN : (h, w, 3)
    # OUT: l → (h, w), a → (h, w), b → (h, w)
    l, a, b = cv2.split(lab)

    # ── Step 3: CLAHE on L channel only ───────────────────────────────────
    # createCLAHE creates the CLAHE processor (not applied yet)
    # clip_limit=2.0  — caps the contrast amplification per tile
    #                   higher = more aggressive, more noise risk
    # tileGridSize=(8,8) — divides image into 8×8 = 64 tiles
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    # clahe.apply runs the equalization on one single-channel image
    # IN : (h, w) — the L channel
    # OUT: (h, w) — L with enhanced local contrast
    l_enhanced = clahe.apply(l)

    # ── Step 4: merge enhanced L back with unchanged A and B ──────────────
    # cv2.merge takes a list of same-shape single-channel arrays
    # and stacks them into one multi-channel array
    # IN : three arrays of shape (h, w)
    # OUT: (h, w, 3) LAB
    lab_enhanced = cv2.merge([l_enhanced, a, b])

    # ── Step 5: LAB → BGR ─────────────────────────────────────────────────
    # IN : (h, w, 3) LAB
    # OUT: (h, w, 3) BGR
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def _binarize(image: np.ndarray,
              block_size: int = 31,
              C: int = 15) -> np.ndarray:
    """
    Convert an image to pure black/white (binary).

    IN  : BGR image (h, w, 3)  OR  grayscale image (h, w)
    OUT : (h, w) uint8 — text = 0 (black), background = 255 (white)

    ── WHY ADAPTIVE, NOT OTSU ───────────────────────────────────────────
    Otsu's method computes ONE global threshold for the entire image.
    A photo taken on a desk with uneven room lighting will have regions
    where the correct threshold differs by 30–50 values. One global
    threshold works well in some areas and fails in others.

    Adaptive Gaussian thresholding computes a DIFFERENT threshold for
    each pixel based on a local neighborhood around it. It handles
    uneven lighting correctly by design.

    ── PARAMETERS ───────────────────────────────────────────────────────
    block_size = 31   — each pixel's threshold is derived from its
                        31×31 pixel neighborhood. Must be an odd number.
    C = 15            — constant subtracted from the computed local mean.
                        Higher C → stricter → sparser result (less black).
                        Tune this if thin pen strokes disappear or noise appears.

    ── STEPS ────────────────────────────────────────────────────────────
    1. Convert to grayscale if input is color
    2. GaussianBlur to reduce noise before thresholding
    3. Adaptive threshold (inverted: text → white, background → black)
    4. Invert back (text → black, background → white)
    """

    # ── Step 1: accept both color and grayscale input ─────────────────────
    # len(image.shape) == 3 means 3 dimensions → color image (h, w, channels)
    # len(image.shape) == 2 means 2 dimensions → already grayscale (h, w)
    if len(image.shape) == 3:
        gray = _to_grayscale(image)
    else:
        gray = image

    # ── Step 2: slight blur to reduce noise impact on thresholding ────────
    # GaussianBlur convolves the image with a Gaussian kernel.
    # (5, 5) — kernel size, must be odd numbers. Larger = more smoothing.
    # 0       — standard deviation in X; 0 means OpenCV computes it from
    #            the kernel size automatically.
    # IN : (h, w) grayscale
    # OUT: (h, w) grayscale, slightly smoothed
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── Step 3: adaptive threshold ────────────────────────────────────────
    # For each pixel: compute the weighted mean of its block_size×block_size
    # neighborhood (Gaussian weights = closer pixels contribute more),
    # subtract C, use that as the threshold for this pixel.
    #
    # THRESH_BINARY_INV: pixels BELOW their local threshold → 255 (white)
    #                    pixels ABOVE their local threshold → 0   (black)
    # This gives us white text on black background (inverted).
    # IN : (h, w) grayscale
    # OUT: (h, w) binary, text=255 (white), background=0 (black)
    binary_inv = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, C
    )

    # ── Step 4: invert to standard document convention ────────────────────
    # Standard convention for document processing: text=0, background=255.
    # cv2.bitwise_not flips every bit in every byte:
    #   0   (00000000) → 255 (11111111)
    #   255 (11111111) → 0   (00000000)
    # IN : (h, w) text=255, background=0
    # OUT: (h, w) text=0,   background=255
    return cv2.bitwise_not(binary_inv)


def _detect_split(binary: np.ndarray,
                  top_clear_ratio: float = 0.08) -> Tuple[bool, int]:
    """
    Determine whether this page has a metadata header, and if so,
    find the row index where the student table begins.

    IN  : binary (h, w) uint8 — full page binary image (text=0, bg=255)
          top_clear_ratio     — fraction of image height to inspect for
                                the presence/absence of table lines
    OUT : (has_header, split_y)
            has_header  bool — True if a metadata header was found
            split_y     int  — row index where the table begins
                               (0 if has_header is False)

    ── DETECTION LOGIC ──────────────────────────────────────────────────
    Key observation from the real sheets:

    PAGE 1 (header page):
        The very top of the image contains the university logo and title
        text — NO table lines exist in the top ~8% of the image height.
        The first table line (top of the "Enseignant" row) appears below
        the logo/title area.

    PAGES 2, 3, 4 (continuation pages):
        The student table starts from the very top edge of the page.
        Table lines appear within the first 1–3% of image height.

    Strategy:
        1. Extract long horizontal lines from the binary image.
        2. Check if any such line exists in the top (top_clear_ratio)
           of the image.
        3. If NO line in the top zone → header page.
           If lines ARE in the top zone → continuation page.
        4. For header pages: find the last long line in the top 50%
           of the image — that is the bottom of the "N° Apo | Nom & Prénom"
           column header row, where the student data begins.

    ── EXTRACTING HORIZONTAL LINES ──────────────────────────────────────
    We use morphological opening with a wide horizontal kernel.
    MORPH_OPEN = erosion then dilation:
        erosion removes everything that doesn't fit the kernel shape
        dilation restores what remains
    With a wide horizontal kernel, only long horizontal lines survive.
    Everything else (text, short marks, vertical lines) is eliminated.
    """
    h, w = binary.shape[:2]

    # ── Extract long horizontal lines ────────────────────────────────────
    # We need white foreground for morphology, but our binary has text=0.
    # bitwise_not inverts: text → 255, background → 0
    foreground = cv2.bitwise_not(binary)

    # Kernel width = 1/15 of image width.
    # A line must be at least this wide to survive the opening operation.
    # This filters out individual characters while preserving printed lines.
    kernel_w = max(1, w // 15)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    h_lines = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, h_kernel)

    # ── Count white pixels per row ────────────────────────────────────────
    # np.sum(h_lines > 0, axis=1):
    #   h_lines > 0  → boolean array: True where white, False where black
    #   np.sum(..., axis=1) → sum along axis 1 (columns) for each row
    #   result: 1D array of length h
    #           each value = number of white pixels in that row
    row_counts = np.sum(h_lines > 0, axis=1)

    # A real table line must span at least 50% of the page width
    line_threshold = w * 0.5
    line_rows = np.where(row_counts > line_threshold)[0]
    # np.where returns a tuple; [0] extracts the array of matching indices

    if len(line_rows) == 0:
        # No horizontal lines detected at all — cannot split
        return False, 0

    # ── Check if the top zone is clear of lines ───────────────────────────
    top_zone_end = int(h * top_clear_ratio)
    lines_in_top_zone = np.sum(line_rows < top_zone_end)

    if lines_in_top_zone > 0:
        # Lines exist near the very top → continuation page, no header
        return False, 0

    # ── Header page: find where the student table begins ─────────────────
    # The split point is the last long horizontal line in the top half
    # of the image — this is the bottom edge of the column header row
    # ("N° Apo | Nom & Prénom | Séance1 | ... | Séance10").
    top_half_lines = line_rows[line_rows < int(h * 0.55)]

    if len(top_half_lines) > 0:
        split_y = int(top_half_lines[-1])
        # [-1] selects the last (bottom-most) matching row
    else:
        # Fallback: assume header occupies top 35%
        split_y = int(h * 0.35)

    return True, split_y


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# Run this file directly to verify preprocessing on a single image.
#
# Usage:
#   python src/preprocessing.py data/raw/real/as1.jpeg
#
# Output:
#   Saves debug_*.jpg files in the current directory for visual inspection.
#   Each file corresponds to one key in the result dictionary.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/preprocessing.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Processing: {path}")

    try:
        result = preprocess(path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"has_header: {result['has_header']}")

    for key, value in result.items():
        if key == 'has_header':
            continue
        if value is None:
            print(f"  {key}: None (continuation page)")
            continue
        filename = f"debug_{key}.jpg"
        cv2.imwrite(filename, value)
        print(f"  {key}: shape={value.shape}  → saved {filename}")

    print("\nOpen the debug_*.jpg files to inspect each preprocessing step.")