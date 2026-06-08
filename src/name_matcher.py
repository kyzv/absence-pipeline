# src/checkbox_detection.py
"""
Stage 4 of the absence pipeline — presence/absence detection.

═══════════════════════════════════════════════════════════════════════════════
INPUTS
═══════════════════════════════════════════════════════════════════════════════
    data/cropped/<doc_id>/as1_table.jpg   ← page 1: starts with column header row
    data/cropped/<doc_id>/as2_table.jpg   ← page 2: starts directly with students
    data/cropped/<doc_id>/as3_table.jpg   ← page 3: starts directly with students
    data/cropped/<doc_id>/as4_table.jpg   ← page 4: starts directly with students
    data/metadata/<doc_id>/metadata.json  ← tells us which séance is active

═══════════════════════════════════════════════════════════════════════════════
TABLE COLUMN LAYOUT (0-based physical column index)
═══════════════════════════════════════════════════════════════════════════════
    Index 0 : N° Apo          ≈ 4% of table width
    Index 1 : Nom & Prénom    ≈ 18% of table width
    Index 2 : Séance 1        ≈ 7.8% of table width
    Index 3 : Séance 2        ≈ 7.8%
    ...
    Index 11: Séance 10       ≈ 7.8%

    active_seance=1 → physical column index = 1 + 1 = 2

═══════════════════════════════════════════════════════════════════════════════
DETECTION METHOD
═══════════════════════════════════════════════════════════════════════════════
    For each student row in the active séance column:
        1. Extract the cell interior (with a small margin to exclude grid lines)
        2. Binarize: text/ink = 0 (black), background = 255 (white)
        3. density = (number of black pixels) / (total pixels in cell)
        4. present if density > density_threshold
           absent  if density ≤ density_threshold

═══════════════════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════════════════
    List of dicts, one per detected student row:
    {
        'global_row'  : int   0-based index across ALL pages of the document
        'page'        : str   filename stem, e.g. 'as1_table'
        'row_in_page' : int   0-based index within that page only
        'present'     : bool  True = signature found, False = cell empty
        'density'     : float measured ink density (use for threshold tuning)
    }
"""

import os
import json
import re
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def detect_absences(doc_id: str,
                    active_seance: Optional[int] = None,
                    cropped_root: str = "data/cropped",
                    metadata_root: str = "data/metadata",
                    density_threshold: float = 0.04,
                    skip_header_row: bool = True,
                    debug: bool = False,
                    debug_root: str = "data/debug") -> List[Dict]:
    """
    Detect presence/absence for every student row across all table pages.

    Args:
        doc_id            : document folder name (e.g. 'doc_1')
        active_seance     : 1-based séance column to check.
                            If None, auto-detected from metadata.json.
        cropped_root      : root folder containing doc subfolders with table images
        metadata_root     : root folder containing doc subfolders with metadata.json
        density_threshold : black pixel fraction above which a cell = present.
                            Start at 0.04, adjust based on debug output.
        skip_header_row   : if True, skip the first row of the first table page.
                            Page 1's table image starts with the column header row
                            ("N° Apo | Nom & Prénom | Séance1...") which is not
                            a student row. Set False if your cropper already
                            excludes it.
        debug             : if True, save annotated images to debug_root.
        debug_root        : folder to write debug images.

    Returns:
        List of dicts — one entry per detected student row in document order.
        Use result['global_row'] to index into the group roster.

    Raises:
        FileNotFoundError if cropped folder or table images are missing.
    """
    # ── 1. Determine which séance column to analyse ───────────────────────
    if active_seance is None:
        active_seance = _get_active_seance(doc_id, metadata_root)
    print(f"[checkbox] doc_id={doc_id}  active_seance={active_seance}  "
          f"threshold={density_threshold}")

    # ── 2. Collect table images in page order ─────────────────────────────
    crop_dir = os.path.join(cropped_root, doc_id)
    if not os.path.isdir(crop_dir):
        raise FileNotFoundError(f"Cropped folder not found: {crop_dir}")

    table_files = _get_sorted_table_files(crop_dir)
    if not table_files:
        raise FileNotFoundError(f"No *_table.jpg images found in: {crop_dir}")

    print(f"[checkbox] Found {len(table_files)} table page(s): "
          f"{[s for s, _ in table_files]}")

    # ── 3. Process each page and accumulate results ───────────────────────
    all_results: List[Dict] = []
    global_offset = 0

    for idx, (page_stem, img_path) in enumerate(table_files):
        is_first = (idx == 0)
        should_skip_header = is_first and skip_header_row

        page_results = _process_one_table(
            img_path        = img_path,
            page_stem       = page_stem,
            active_seance   = active_seance,
            density_threshold = density_threshold,
            global_offset   = global_offset,
            skip_header_row = should_skip_header,
            debug           = debug,
            debug_root      = debug_root,
        )

        all_results.extend(page_results)
        global_offset += len(page_results)
        print(f"[checkbox]   {page_stem}: {len(page_results)} rows detected")

    present_count = sum(1 for r in all_results if r['present'])
    absent_count  = len(all_results) - present_count
    print(f"[checkbox] Total: {len(all_results)} rows — "
          f"{present_count} present, {absent_count} absent")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE SÉANCE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _get_active_seance(doc_id: str, metadata_root: str) -> int:
    """
    Read metadata.json and return the 1-based index of the first séance
    that has at least one non-empty metadata field (date, heure_debut, etc.).
    Falls back to 1 if the file is missing or all séances are empty.

    IN  : doc_id, path to metadata root
    OUT : int — 1-based séance number

    The metadata.json structure:
        {
          "seances": {
            "1": {"date": "04/06/26", "heure_debut": "08h30", ...},
            "2": {"date": "", ...},
            ...
          }
        }
    The first séance with a non-empty value is the active one.
    """
    meta_path = os.path.join(metadata_root, doc_id, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"[checkbox] metadata.json not found, defaulting to séance 1")
        return 1

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    # sorted() on string keys "1".."10": sort by integer value
    for num_str, info in sorted(meta.get('seances', {}).items(),
                                 key=lambda kv: int(kv[0])):
        # info is a dict like {"date": "04/06/26", "heure_debut": "08h30", ...}
        # any() returns True if at least one value is a non-empty string
        if any(str(v).strip() for v in info.values() if v):
            return int(num_str)

    print(f"[checkbox] All séances empty in metadata, defaulting to 1")
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def _get_sorted_table_files(crop_dir: str) -> List[Tuple[str, str]]:
    """
    Return [(stem, full_path), ...] for table images, sorted by page number.

    IN  : crop_dir — path to data/cropped/<doc_id>/
    OUT : sorted list of (stem, path) tuples
          stem example: "as1_table"
          path example: "data/cropped/doc_1/as1_table.jpg"

    Sort key: the integer N extracted from "asN_table.*".
    Non-matching files are assigned sort key 0 and sorted first.
    """
    def _page_num(fname: str) -> int:
        m = re.search(r'as(\d+)', fname, re.IGNORECASE)
        return int(m.group(1)) if m else 0

    files = [
        f for f in os.listdir(crop_dir)
        if re.search(r'_table\.(jpg|jpeg|png)$', f, re.IGNORECASE)
    ]
    files.sort(key=_page_num)

    return [
        (os.path.splitext(f)[0], os.path.join(crop_dir, f))
        for f in files
    ]


# ─────────────────────────────────────────────────────────────────────────────
# PER-PAGE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _process_one_table(img_path: str,
                        page_stem: str,
                        active_seance: int,
                        density_threshold: float,
                        global_offset: int,
                        skip_header_row: bool,
                        debug: bool,
                        debug_root: str) -> List[Dict]:
    """
    Process a single table image page and return per-row presence results.

    IN  : img_path        — path to the cropped table JPEG
          page_stem       — filename stem for labelling output
          active_seance   — 1-based séance number to check
          density_threshold — ink fraction threshold for presence
          global_offset   — number of student rows already processed
          skip_header_row — if True, skip the first detected row
          debug/debug_root — optional debug image output

    OUT : list of row result dicts
    """
    # ── Load ──────────────────────────────────────────────────────────────
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    binary = _binarize(image)
    h, w   = binary.shape

    # ── Detect grid structure ─────────────────────────────────────────────
    row_bounds = _find_row_boundaries(binary)
    col_bounds = _find_col_boundaries(binary)

    # ── Identify the target séance column ─────────────────────────────────
    # Physical column index: 0=N°Apo, 1=Nom&Prénom, 2=Séance1, 3=Séance2...
    # active_seance=1 → index 2, active_seance=2 → index 3, etc.
    target_col_idx = active_seance + 1
    seance_x1, seance_x2 = _get_col_bounds(col_bounds, target_col_idx, w)

    # ── Build row strip list ───────────────────────────────────────────────
    # A "row strip" is the vertical span (y1, y2) of one table row.
    # Consecutive boundary pairs define the strips.
    row_strips = []
    for i in range(len(row_bounds) - 1):
        y1 = row_bounds[i]
        y2 = row_bounds[i + 1]
        row_h = y2 - y1
        # Valid student rows have a height in a plausible range.
        # Very thin strips (< 8px) are noise between adjacent detected lines.
        # Very tall strips (> 150px) are likely image artifacts.
        if 8 <= row_h <= 150:
            row_strips.append((y1, y2))

    # ── Skip column header row if needed ─────────────────────────────────
    # Page 1's table starts with a printed column header row:
    # "N° Apo | Nom & Prénom | Séance1 | Séance2 | ..."
    # This is NOT a student row. Skip it.
    if skip_header_row and row_strips:
        row_strips = row_strips[1:]

    # ── Compute ink density for each row's séance cell ───────────────────
    MARGIN = 4          # pixels to trim from each edge to exclude grid lines
    results: List[Dict] = []

    for strip_idx, (y1, y2) in enumerate(row_strips):
        # Clip to image bounds then apply margin
        cy1 = max(0, y1 + MARGIN)
        cy2 = max(0, y2 - MARGIN)
        cx1 = max(0, seance_x1 + MARGIN)
        cx2 = max(0, seance_x2 - MARGIN)

        cell = binary[cy1:cy2, cx1:cx2]

        if cell.size == 0:
            density = 0.0
        else:
            # binary convention: ink = 0 (black), background = 255 (white)
            # np.sum(cell < 128) counts every black/near-black pixel
            black_pixels = int(np.sum(cell < 128))
            density = black_pixels / cell.size

        results.append({
            'global_row':  global_offset + len(results),
            'page':        page_stem,
            'row_in_page': strip_idx,
            'present':     density > density_threshold,
            'density':     round(float(density), 5),
        })

    # ── Debug output ──────────────────────────────────────────────────────
    if debug:
        _save_debug_image(image, row_strips, seance_x1, seance_x2,
                          results, debug_root, page_stem)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _binarize(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR or grayscale image to a binary image.

    IN  : (h, w, 3) uint8  BGR
          OR (h, w) uint8  grayscale
    OUT : (h, w) uint8  — ink/text = 0 (black), background = 255 (white)

    Uses adaptive Gaussian thresholding which computes a different threshold
    per pixel based on a local neighborhood. This handles uneven lighting
    better than a global threshold (Otsu).

    block_size=31 — neighborhood size (must be odd). 31×31 pixels per tile.
    C=15          — constant subtracted from local mean. Higher = sparser.
    """
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # THRESH_BINARY_INV: below threshold → 255 (text=white), above → 0 (bg=black)
    binary_inv = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 15
    )
    # bitwise_not: 0 ↔ 255  →  text = 0 (black), background = 255 (white)
    return cv2.bitwise_not(binary_inv)


def _find_row_boundaries(binary: np.ndarray,
                          min_span_ratio: float = 0.35) -> List[int]:
    """
    Detect y-coordinates of horizontal table lines.

    IN  : binary (h, w) uint8 — text=0, background=255
          min_span_ratio      — line must span at least this fraction of width
    OUT : sorted list of y-coordinate integers including 0 and h as endpoints

    ── HOW IT WORKS ─────────────────────────────────────────────────────────
    1. Invert the binary image so grid lines become white (255) on black (0).
       Morphological operations treat white pixels as "foreground".
    2. Apply MORPH_OPEN with a wide horizontal kernel.
       The kernel is (min_span * width) pixels wide and 1 pixel tall.
       Opening = erosion then dilation:
         - Erosion removes everything that doesn't extend at least kernel_w pixels
         - Dilation restores what remains
       Net effect: only long horizontal structures survive.
       Short marks, text, noise are eliminated.
    3. Count surviving white pixels per row.
    4. Find rows exceeding the threshold (= line locations).
    5. Group consecutive detected rows into single boundary positions.
       Multiple adjacent rows may light up for one actual line due to
       the line's finite thickness (2-3px). _group_consecutive merges them.
    6. Add 0 and h as first/last boundaries so all rows are enclosed.
    """
    h, w = binary.shape
    inv  = cv2.bitwise_not(binary)

    # Kernel: wide enough to reject noise, narrow enough to catch real lines
    kernel_w = max(10, int(w * min_span_ratio))
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    h_lines  = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)

    # Count white pixels per row (axis=1 sums across columns for each row)
    row_sums  = np.sum(h_lines > 0, axis=1)
    # A real line should illuminate most of the min_span region
    threshold = w * min_span_ratio * 0.7
    detected  = np.where(row_sums > threshold)[0]

    if len(detected) == 0:
        return [0, h]

    boundaries = _group_consecutive(detected, gap=6)

    # Ensure the list starts at 0 and ends at h
    if boundaries[0] > 5:
        boundaries = [0] + boundaries
    if boundaries[-1] < h - 5:
        boundaries = boundaries + [h]

    return boundaries


def _find_col_boundaries(binary: np.ndarray,
                          min_span_ratio: float = 0.20) -> List[int]:
    """
    Detect x-coordinates of vertical table lines.

    IN  : binary (h, w) uint8 — text=0, background=255
          min_span_ratio      — line must span at least this fraction of height
    OUT : sorted list of x-coordinate integers including 0 and w as endpoints

    Identical logic to _find_row_boundaries but transposed:
    uses a tall vertical kernel (1 × kernel_h) to isolate vertical lines.
    """
    h, w = binary.shape
    inv  = cv2.bitwise_not(binary)

    kernel_h = max(10, int(h * min_span_ratio))
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    v_lines  = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)

    col_sums  = np.sum(v_lines > 0, axis=0)   # sum along rows for each column
    threshold = h * min_span_ratio * 0.7
    detected  = np.where(col_sums > threshold)[0]

    if len(detected) == 0:
        return [0, w]

    boundaries = _group_consecutive(detected, gap=6)

    if boundaries[0] > 5:
        boundaries = [0] + boundaries
    if boundaries[-1] < w - 5:
        boundaries = boundaries + [w]

    return boundaries


def _group_consecutive(positions: np.ndarray, gap: int = 5) -> List[int]:
    """
    Merge runs of close integer positions into single representative values.

    IN  : positions — sorted array of pixel indices (e.g. row/col numbers)
          gap       — max separation to still be considered the same line
    OUT : list of ints — one centroid per detected line

    Example:
        positions = [100, 101, 102, 108, 109, 200]  gap=5
        groups    = [[100,101,102], [108,109], [200]]
        centroids = [101,           108,        200]
    """
    if len(positions) == 0:
        return []

    groups  = []
    current = [int(positions[0])]

    for pos in positions[1:]:
        if int(pos) - current[-1] <= gap:
            current.append(int(pos))
        else:
            groups.append(int(np.mean(current)))
            current = [int(pos)]

    groups.append(int(np.mean(current)))
    return groups


def _get_col_bounds(col_boundaries: List[int],
                    target_col_idx: int,
                    image_width: int) -> Tuple[int, int]:
    """
    Return (x_start, x_end) for the column at target_col_idx.

    Tries to use detected col_boundaries first.
    Falls back to known document template ratios if detection is incomplete.

    IN  : col_boundaries  — list of detected x-positions (from _find_col_boundaries)
          target_col_idx  — 0-based physical column index
          image_width     — full image width in pixels
    OUT : (x1, x2) pixel coordinates of the column's left and right edges

    ── COLUMN LAYOUT (document template) ────────────────────────────────────
    These ratios are measured from the real absence sheets:
        Index 0 : N° Apo           0%  –  5%  of width
        Index 1 : Nom & Prénom     5%  – 23%  of width
        Index 2 : Séance 1        23%  – 30.8%
        Index 3 : Séance 2        30.8%– 38.5%
        Index k : Séance (k-1)    (5% + 18%) + (k-2) × 7.7%  to  same + 7.7%

    These ratios assume a fully visible table with no perspective distortion.
    They are used only as fallback when vertical line detection is unreliable.
    """
    # ── Primary: use detected boundaries ─────────────────────────────────
    if len(col_boundaries) >= target_col_idx + 2:
        x1 = col_boundaries[target_col_idx]
        x2 = col_boundaries[target_col_idx + 1]
        if x2 - x1 >= 15:   # sanity check: column must be at least 15px wide
            return x1, x2

    # ── Fallback: template ratios ─────────────────────────────────────────
    W = image_width

    # Fixed columns
    COL_NAPO_END  = 0.05
    COL_NOM_END   = 0.23
    COL_SEANCE_W  = 0.077        # width of each séance column

    if target_col_idx == 0:
        return 0, int(W * COL_NAPO_END)

    if target_col_idx == 1:
        return int(W * COL_NAPO_END), int(W * COL_NOM_END)

    # target_col_idx >= 2 → séance column
    seance_num = target_col_idx - 1        # 1-based séance number
    x1 = int(W * (COL_NOM_END + (seance_num - 1) * COL_SEANCE_W))
    x2 = int(W * (COL_NOM_END +  seance_num      * COL_SEANCE_W))
    return x1, x2


# ─────────────────────────────────────────────────────────────────────────────
# DEBUG OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def _save_debug_image(image: np.ndarray,
                      row_strips: List[Tuple[int, int]],
                      seance_x1: int,
                      seance_x2: int,
                      results: List[Dict],
                      debug_root: str,
                      page_stem: str) -> None:
    """
    Save an annotated version of the table image for visual verification.

    Annotations:
        Blue  horizontal lines  — detected row boundaries
        Orange vertical lines   — boundaries of the analysed séance column
        Green  cell overlay     — present (signed)
        Red    cell overlay     — absent (unsigned)
        Density value           — printed inside each cell

    IN  : image       — original BGR table image
          row_strips  — list of (y1, y2) tuples for each student row
          seance_x1/2 — left/right pixel bounds of the séance column
          results     — list of row result dicts
          debug_root  — folder to save the output image
          page_stem   — used as the output filename
    """
    os.makedirs(debug_root, exist_ok=True)
    dbg = image.copy()
    h_img = dbg.shape[0]

    # Draw séance column boundaries
    # (0, 165, 255) = orange in BGR
    cv2.line(dbg, (seance_x1, 0), (seance_x1, h_img), (0, 165, 255), 2)
    cv2.line(dbg, (seance_x2, 0), (seance_x2, h_img), (0, 165, 255), 2)

    for i, (y1, y2) in enumerate(row_strips):
        # Row boundary line (blue)
        cv2.line(dbg, (0, y1), (dbg.shape[1], y1), (255, 0, 0), 1)

        if i >= len(results):
            continue

        # Cell highlight (green = present, red = absent)
        color = (0, 200, 0) if results[i]['present'] else (0, 0, 200)
        overlay = dbg.copy()
        cv2.rectangle(overlay, (seance_x1, y1), (seance_x2, y2), color, cv2.FILLED)
        # alpha = 0.35: 35% overlay color, 65% original image
        cv2.addWeighted(overlay, 0.35, dbg, 0.65, 0, dbg)
        cv2.rectangle(dbg, (seance_x1, y1), (seance_x2, y2), color, 1)

        # Density label inside cell
        label = f"{results[i]['density']:.3f}"
        cv2.putText(dbg, label,
                    (seance_x1 + 2, y1 + (y2 - y1) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1,
                    cv2.LINE_AA)

    out_path = os.path.join(debug_root, f"{page_stem}_debug.jpg")
    cv2.imwrite(out_path, dbg)
    print(f"[checkbox]   Debug image saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND-LINE TEST
#
# Usage:
#   python src/checkbox_detection.py doc_1
#   python src/checkbox_detection.py doc_1 1 0.04
#   python src/checkbox_detection.py doc_1 1 0.04 debug
#
# This prints all detected rows with their density values.
# Run this first with debug=True to visually verify:
#   1. Row boundaries are detected correctly (blue lines in debug image)
#   2. The séance column is in the right place (orange lines)
#   3. Density values are clearly separated (present >> threshold > absent)
# Then adjust density_threshold accordingly.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/checkbox_detection.py <doc_id> "
              "[seance] [threshold] [debug]")
        sys.exit(1)

    _doc_id    = sys.argv[1]
    _seance    = int(sys.argv[2])   if len(sys.argv) > 2 else None
    _threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.04
    _debug     = (sys.argv[4].lower() == "debug") if len(sys.argv) > 4 else False

    _results = detect_absences(
        doc_id            = _doc_id,
        active_seance     = _seance,
        density_threshold = _threshold,
        debug             = _debug,
    )

    # Print summary
    print()
    print(f"{'Row':>4}  {'Page':<14}  {'PgRow':>5}  {'Status':<8}  {'Density':>8}")
    print("-" * 52)
    for r in _results:
        status = "PRESENT" if r['present'] else "absent "
        print(f"{r['global_row']:>4}  {r['page']:<14}  "
              f"{r['row_in_page']:>5}  {status}  {r['density']:>8.5f}")

    print()
    present = sum(1 for r in _results if r['present'])
    absent  = len(_results) - present
    print(f"Total: {len(_results)}   Present: {present}   Absent: {absent}")