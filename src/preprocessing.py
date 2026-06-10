"""
src/preprocessing.py
--------------------
Step 1 of the pipeline: prepare a raw scanned absence sheet image for analysis.

What this module does, in order:
  1. Load the image from disk.
  2. Fix orientation  – if the scan is portrait (h > w), rotate it to landscape.
  3. Enhance contrast – CLAHE so ink (red/blue handwriting) pops against the background.
  4. Detect the table grid lines using morphological operations (long horizontal/vertical kernels).
  5. Deskew – measure the tilt of those lines via HoughLinesP and correct it.
  6. Tight-crop – project the grid-line pixels onto each axis and cut exactly to where lines exist.

Output: a clean, straight, margin-free image ready for OCR and grid analysis.
"""

import cv2
import numpy as np
import re


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare(image_path: str) -> np.ndarray:
    """
    Full preprocessing pipeline for one page image.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the raw scanned image (.jpg / .png).

    Returns
    -------
    np.ndarray
        A BGR image that is landscape-oriented, contrast-enhanced,
        deskewed, and cropped to the table boundary.
    """
    img = _load(image_path)
    img = _fix_orientation(img)
    img = _enhance_contrast(img)
    img = _deskew(img)
    img = _crop_to_grid(img)
    return img


def save_debug(image: np.ndarray, path: str) -> None:
    """Save an intermediate image to disk for visual inspection."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, image)


def natural_sort_key(filename: str):
    """Sort key that orders 'as2.jpg' before 'as10.jpg'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', filename)]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load(path: str) -> np.ndarray:
    """Load image from disk and raise a clear error if it fails."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _fix_orientation(img: np.ndarray) -> np.ndarray:
    """
    Absence sheets are landscape.  If the scanner saved them as portrait
    (height > width), rotate 90° counter-clockwise to restore landscape layout.
    """
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) improves local
    contrast.  Working in LAB color space lets us enhance only the luminance
    channel (L) without shifting colors, so red/blue handwriting stays vivid.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _build_line_kernels(img: np.ndarray):
    """
    Return (h_kernel, v_kernel) sized to detect the full-width table lines
    while ignoring shorter strokes like individual letters.

    The horizontal kernel is 150 px wide — only a line spanning at least 150 px
    survives MORPH_OPEN.  The vertical kernel is 100 px tall for the same reason.
    We use slightly smaller kernels here so that grid lines broken by signatures
    are not completely erased.
    """
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    return h_kernel, v_kernel


def _extract_lines(gray: np.ndarray, h_kernel, v_kernel):
    """
    Binarize and apply morphological OPEN with the two kernels to isolate
    long horizontal lines and long vertical lines separately, then combine them.
    """
    # THRESH_BINARY_INV + OTSU: ink becomes white (255), background becomes black (0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Small dilation closes tiny gaps so lines are continuous
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    h_lines = cv2.dilate(h_lines, dilate_k)
    v_lines = cv2.dilate(v_lines, dilate_k)

    return h_lines, v_lines, cv2.add(h_lines, v_lines)


def _deskew(img: np.ndarray) -> np.ndarray:
    """
    Measure the median angle of the detected horizontal table lines and rotate
    the image so they become perfectly horizontal.

    Uses Canny edge detection instead of morphological lines to find the true
    skew angle, because morphological kernels force pixels to be horizontal
    and destroy sub-degree skew information.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200,
                            minLineLength=300, maxLineGap=50)
    angle = 0.0
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # We only care about lines that are roughly horizontal
            if -15 < a < 15 and abs(a) > 0.1:
                angles.append(a)
        if angles:
            angle = float(np.median(angles))

    if abs(angle) < 0.05:   # Already straight — skip warp
        return img

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))


def _crop_to_grid(img: np.ndarray) -> np.ndarray:
    """
    Crop the image tightly to the table content using a hybrid strategy:

    - LEFT / RIGHT  : use detected vertical grid lines.
                      These are always present across every page (including
                      continuation pages) so they give reliable column boundaries.

    - TOP / BOTTOM  : use content detection (first/last non-white pixel row).
                      Continuation pages have no top horizontal border above the
                      first student row.  Relying on grid lines here would cut
                      that first row off.  Scanning for actual ink is safer.

    A small outward margin (15 px) is kept on each side.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_kernel, v_kernel = _build_line_kernels(img)
    _, v_lines, _ = _extract_lines(gray, h_kernel, v_kernel)

    # ---- LEFT / RIGHT from vertical lines -----------------------------------
    col_proj = np.sum(v_lines, axis=0)          # sum each column
    col_idx  = np.where(col_proj > 0)[0]

    if len(col_idx) == 0:
        return img                              # no lines found — give up

    margin = 15
    x0 = max(0,             col_idx[0]  - margin)
    x1 = min(img.shape[1],  col_idx[-1] + margin)

    # ---- TOP / BOTTOM -------------------------------------------------------
    # TOP  → first content row (catches pages with no top border line)
    # BOTTOM → last detected horizontal grid line + small padding
    #          (ignores blank white space below the table that may contain
    #           a stray teacher name / signature written outside the grid)

    h_lines_only, _, _ = _extract_lines(gray, h_kernel, v_kernel)

    # Top: first non-white row within the column strip
    strip = gray[:, x0:x1]
    content_mask = strip < 240
    row_has_content = content_mask.any(axis=1)
    content_rows = np.where(row_has_content)[0]

    if len(content_rows) == 0:
        return img

    y_top = content_rows[0]

    # Bottom: last row that has a long horizontal line
    h_row_proj = np.sum(h_lines_only, axis=1)
    h_row_idx  = np.where(h_row_proj > 0)[0]

    if len(h_row_idx) > 0:
        y_bottom = h_row_idx[-1]
        # Allow up to 80 px of content below the last grid line
        # (e.g. EMARGEMENT row, teacher name written just below the border)
        extra_content = content_rows[content_rows > y_bottom]
        if len(extra_content) > 0 and extra_content[-1] - y_bottom < 80:
            y_bottom = extra_content[-1]
    else:
        # Fallback if no h-lines found: use last content row
        y_bottom = content_rows[-1]

    y0 = max(0,             y_top    - margin)
    y1 = min(img.shape[0],  y_bottom + margin)

    return img[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python src/preprocessing.py <doc_id>")
        print("Example: python src/preprocessing.py doc_2")
        sys.exit(1)

    doc_id   = sys.argv[1]
    raw_dir  = os.path.join("data", "raw", doc_id)
    out_dir  = os.path.join("data", "preprocessed", doc_id)

    if not os.path.isdir(raw_dir):
        print(f"ERROR: Directory not found: {raw_dir}")
        sys.exit(1)

    exts = ('.jpg', '.jpeg', '.png')
    pages = sorted(
        [f for f in os.listdir(raw_dir) if f.lower().endswith(exts)],
        key=natural_sort_key
    )

    if not pages:
        print(f"No images found in {raw_dir}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    for fname in pages:
        src_path = os.path.join(raw_dir, fname)
        dst_path = os.path.join(out_dir, fname)
        print(f"  Processing {fname} ...", end=" ", flush=True)
        result = prepare(src_path)
        cv2.imwrite(dst_path, result)
        print(f"saved -> {dst_path}  (shape {result.shape[1]}x{result.shape[0]})")

    print(f"\nDone. {len(pages)} page(s) saved to {out_dir}")