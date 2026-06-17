<<<<<<< HEAD

=======
"""
src/preprocessing.py
--------------------

  1. Load the image from disk.
  2. Fix orientation  – if the scan is portrait (h > w), rotate it to landscape.
  3. Enhance contrast – CLAHE so ink (red/blue handwriting) pops against the background.
  4. Detect the table grid lines using morphological operations (long horizontal/vertical kernels).
  5. Deskew – measure the tilt of those lines via HoughLinesP and correct it.
  6. Tight-crop – project the grid-line pixels onto each axis and cut exactly to where lines exist.

"""
>>>>>>> c7c161267fc4489ac10746773a77ef47e58bc3d2

import cv2
import numpy as np
import re



def prepare(image_path: str) -> np.ndarray:
<<<<<<< HEAD
    
=======
    """
    
    """
>>>>>>> c7c161267fc4489ac10746773a77ef47e58bc3d2
    img = _load(image_path)
    img = _fix_orientation(img)
    img = _enhance_contrast(img)
    img = _deskew(img)
    img = _crop_to_grid(img)
    return img


def save_debug(image: np.ndarray, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, image)


def natural_sort_key(filename: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', filename)]



def _load(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _fix_orientation(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _enhance_contrast(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _build_line_kernels(img: np.ndarray):
<<<<<<< HEAD
=======
    """
    Return (h_kernel, v_kernel) sized to detect the full-width table lines
    while ignoring shorter strokes like individual letters.

    The horizontal kernel is 150 px wide — only a line spanning at least 150 px
    survives MORPH_OPEN.  The vertical kernel is 100 px tall for the same reason.
     uses slightly smaller kernels here so that grid lines broken by signatures
    are not completely erased.
    """
>>>>>>> c7c161267fc4489ac10746773a77ef47e58bc3d2
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    return h_kernel, v_kernel


def _extract_lines(gray: np.ndarray, h_kernel, v_kernel):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    h_lines = cv2.dilate(h_lines, dilate_k)
    v_lines = cv2.dilate(v_lines, dilate_k)

    return h_lines, v_lines, cv2.add(h_lines, v_lines)


def _deskew(img: np.ndarray) -> np.ndarray:
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    angles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 300:  # Only consider long horizontal lines
            [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            a = np.degrees(np.arctan2(vy, vx))
            # We only care about lines that are roughly horizontal
            if -15 < a < 15:
                angles.append(a[0])
                
    angle = float(np.median(angles)) if angles else 0.0

    if abs(angle) < 0.05:
        return img

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))


def _crop_to_grid(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_kernel, v_kernel = _build_line_kernels(img)
    _, v_lines, _ = _extract_lines(gray, h_kernel, v_kernel)

   
    col_proj = np.sum(v_lines, axis=0)         
    col_idx  = np.where(col_proj > 0)[0]

    if len(col_idx) == 0:
        return img                              

    margin = 15
    x0 = max(0,             col_idx[0]  - margin)
    x1 = min(img.shape[1],  col_idx[-1] + margin)

   
    h_lines_only, _, _ = _extract_lines(gray, h_kernel, v_kernel)

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
