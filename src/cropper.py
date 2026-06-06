# src/cropper.py
"""
Stage 2 of the absence pipeline -/  region separation.

Responsibilities:
    1. Load a preprocessed image (output of preprocessing.py).
    2. Detect the grey table‑header row ( "N° Apo | Nom & Prénom" ) that
       separates the document into metadata header and student table.
    3. If no grey row is found and the page is the first of the document,
       fall back to a fixed percentage split (top 30%).
    4. If the page is a continuation (not first), treat the entire image
       as the table – no header region exists.
    5. Save header and table images to data/cropped/ and return their paths
       (or None) for the next stage.

This module does NOT perform OCR or checkbox detection.

Input  : path to a preprocessed image (e.g., data/preprocessed/as1.jpeg)
         + a flag indicating whether this is the first page of the document
Output : header image saved to data/cropped/{base}_header.jpg
         table  image saved to data/cropped/{base}_table.jpg
         + dict with keys 'header_path' (or None),   'table_path'
"""

import os
import cv2
import numpy as np
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def crop_page(image_path: str,
              is_first_page: bool = False,
              output_dir: str = "data/cropped") -> Dict[str, Optional[str]]:
    """
    Split one preprocessed page into header and table regions.

    Args:
        image_path   : path to a preprocessed image (cleaned, BGR).
        is_first_page: True if this image is the first page of the absence
                       document (the one that may contain the metadata header).
        output_dir   : folder where cropped images will be saved.
                       Created automatically.

    Returns:
        {
            'has_header'  : bool,
            'header_path' : str or None,
            'table_path'  : str
        }
    """
    image = _load(image_path)
    h, w = image.shape[:2]

    # ── Detection of the grey separator row ──────────────────────────────
    # We only look for it on the first page.
    if is_first_page:
        split_y = _detect_grey_row(image)
        if split_y is None:
            # Detection failed → use fallback (top 30% of height)
            split_y = int(h * 0.30)
        # Safety: never cut at extreme edge
        split_y = max(30, min(h - 10, split_y))
    else:
        # Continuation pages have no header – the whole image is the table.
        split_y = 0

    # ── Crop ─────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    if split_y > 0:
        header_img = image[:split_y, :]
        table_img  = image[split_y:, :]

        header_path = os.path.join(output_dir, f"{base}_header.jpg")
        table_path  = os.path.join(output_dir, f"{base}_table.jpg")
        cv2.imwrite(header_path, header_img)
        cv2.imwrite(table_path,  table_img)
        return {
            'has_header': True,
            'header_path': header_path,
            'table_path': table_path
        }
    else:
        # No header region – save whole image as table
        table_path = os.path.join(output_dir, f"{base}_table.jpg")
        cv2.imwrite(table_path, image)
        return {
            'has_header': False,
            'header_path': None,
            'table_path': table_path
        }


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: str) -> np.ndarray:
    """
    Load a BGR image from disk. Raises FileNotFoundError on failure.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _detect_grey_row(image: np.ndarray) -> Optional[int]:
    """
    Find the y‑coordinate of the grey table‑header row.

    ── APPROACH ──────────────────────────────────────────────────────────
    The row we want has three characteristics:
        1. It is a long horizontal band – almost the entire image width.
        2. Its pixels are grey (R ≈ G ≈ B) with a medium intensity
           (not white, not black).
        3. Because it contains the printed text "N° Apo | Nom & Prénom",
           some black pixels are embedded, but they are narrow vertical
           interruptions. The bulk of the row is a smooth grey background.

    We compute for every pixel whether it is “likely grey”:
        - The three colour channels must be close to each other
          (max - min < 25).  This rejects coloured pixels (logo, etc.).
        - The average brightness (R+G+B)/3 must be between 100 and 200.
          This rejects pure white (near 255) and dark black text (near 0).

    Then we sum the number of “grey” pixels per row.
    The row with the highest count that also spans a significant fraction
    of the image width (≥ 70%) is chosen as the separator.

    Returns the row index, or None if no suitable row is found.
    """
    # We work on a float copy to avoid integer overflows (not strictly
    # necessary with uint8, but keeps calculations clean).
    bgr = image.astype(np.float32)

    # ── Step 1: identify grey pixels ────────────────────────────────────
    # Channel differences: max(R,G,B) - min(R,G,B) along axis=2 (colour axis)
    min_vals = np.min(bgr, axis=2)          # (h, w) – darkest channel per pixel
    max_vals = np.max(bgr, axis=2)          # (h, w) – brightest channel
    diff     = max_vals - min_vals          # (h, w) – colour variation

    # Mean brightness (R+G+B)/3
    mean_bright = np.mean(bgr, axis=2)      # (h, w)

    # Boolean mask: True where pixel is grey
    is_grey = (diff < 25) & (mean_bright > 100) & (mean_bright < 200)

    # ── Step 2: count grey pixels per row ───────────────────────────────
    grey_counts = np.sum(is_grey, axis=1)   # (h,) – number of grey pixels per row

    # We require a row to have at least 70% of the image width as grey
    width_threshold = image.shape[1] * 0.70
    candidate_rows = np.where(grey_counts > width_threshold)[0]

    if len(candidate_rows) == 0:
        return None

    # ── Step 3: pick the row with the absolute highest grey‑pixel count ─
    # Among the candidates, the one with the largest count is likely the
    # true grey bar, because it is the most uninterrupted.
    best_row = candidate_rows[np.argmax(grey_counts[candidate_rows])]

    # ── Optional: we can refine by looking for the centre of the band ───
    # The grey bar is several pixels tall. We’ll return the middle of the
    # contiguous block of rows that pass the threshold around best_row.
    # This helps avoid cutting exactly on the top edge.
    # Find the vertical extent:
    top = best_row
    while top > 0 and grey_counts[top - 1] > width_threshold:
        top -= 1
    bottom = best_row
    while bottom < image.shape[0] - 1 and grey_counts[bottom + 1] > width_threshold:
        bottom += 1

    # Return the row just below the grey bar (so header is above, table below)
    return bottom + 1


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND‑LINE TEST
# Usage: python src/cropper.py data/preprocessed/as1.jpeg --first
#        python src/cropper.py data/preprocessed/as2.jpeg
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/cropper.py <preprocessed_image> [--first]")
        sys.exit(1)

    img_path = sys.argv[1]
    first = "--first" in sys.argv

    result = crop_page(img_path, is_first_page=first)

    print(f"Has header? {result['has_header']}")
    if result['header_path']:
        print(f"Header saved → {result['header_path']}")
    print(f"Table saved  → {result['table_path']}")