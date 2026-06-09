# src/checkbox_detection.py
import os
import json
import re
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

def detect_absences(doc_id: str,
                    active_seance: Optional[int] = None,
                    cropped_root: str = "data/cropped",
                    metadata_root: str = "data/metadata",
                    density_threshold: float = 0.04,
                    skip_header_row: bool = True,
                    debug: bool = False,
                    debug_root: str = "data/debug") -> List[Dict]:
    if active_seance is None:
        active_seance = _get_active_seance(doc_id, metadata_root)
    print(f"[checkbox] doc_id={doc_id}  active_seance={active_seance}  "
          f"threshold={density_threshold}")

    crop_dir = os.path.join(cropped_root, doc_id)
    if not os.path.isdir(crop_dir):
        raise FileNotFoundError(f"Cropped folder not found: {crop_dir}")

    table_files = _get_sorted_table_files(crop_dir)
    if not table_files:
        raise FileNotFoundError(f"No *_table.jpg images found in: {crop_dir}")

    print(f"[checkbox] Found {len(table_files)} table page(s): {[s for s, _ in table_files]}")

    all_results: List[Dict] = []
    global_offset = 0

    for idx, (page_stem, img_path) in enumerate(table_files):
        is_first = (idx == 0)
        should_skip_header = is_first and skip_header_row

        page_results = _process_one_table(
            img_path=img_path,
            page_stem=page_stem,
            active_seance=active_seance,
            density_threshold=density_threshold,
            global_offset=global_offset,
            skip_header_row=should_skip_header,
            debug=debug,
            debug_root=debug_root,
        )

        all_results.extend(page_results)
        global_offset += len(page_results)
        print(f"[checkbox]   {page_stem}: {len(page_results)} rows detected")

    present_count = sum(1 for r in all_results if r['present'])
    absent_count = len(all_results) - present_count
    print(f"[checkbox] Total: {len(all_results)} rows — {present_count} present, {absent_count} absent")

    return all_results

def _get_active_seance(doc_id: str, metadata_root: str) -> int:
    meta_path = os.path.join(metadata_root, doc_id, "metadata.json")
    if not os.path.exists(meta_path):
        return 1
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    for num_str, info in sorted(meta.get('seances', {}).items(), key=lambda kv: int(kv[0])):
        if any(str(v).strip() for v in info.values() if v):
            return int(num_str)
    return 1

def _get_sorted_table_files(crop_dir: str) -> List[Tuple[str, str]]:
    def _page_num(fname: str) -> int:
        m = re.search(r'as(\d+)', fname, re.IGNORECASE)
        return int(m.group(1)) if m else 0
    files = [f for f in os.listdir(crop_dir) if re.search(r'_table\.(jpg|jpeg|png)$', f, re.IGNORECASE)]
    files.sort(key=_page_num)
    return [(os.path.splitext(f)[0], os.path.join(crop_dir, f)) for f in files]

def _process_one_table(img_path: str, page_stem: str, active_seance: int, density_threshold: float,
                       global_offset: int, skip_header_row: bool, debug: bool, debug_root: str) -> List[Dict]:
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    binary = _binarize(image)
    h, w = binary.shape

    row_bounds = _find_row_boundaries(binary)
    col_bounds = _find_col_boundaries(binary)

    seance_x1, seance_x2 = _get_target_col_bounds(col_bounds, active_seance, w)
    row_strips = _get_robust_row_strips(row_bounds)

    if skip_header_row and row_strips:
        row_strips = row_strips[1:]

    MARGIN = 4
    results: List[Dict] = []

    for strip_idx, (y1, y2) in enumerate(row_strips):
        cy1 = max(0, y1 + MARGIN)
        cy2 = max(0, y2 - MARGIN)
        cx1 = max(0, seance_x1 + MARGIN)
        cx2 = max(0, seance_x2 - MARGIN)

        cell = binary[cy1:cy2, cx1:cx2]
        if cell.size == 0:
            density = 0.0
        else:
            black_pixels = int(np.sum(cell < 128))
            density = black_pixels / cell.size

        results.append({
            'global_row': global_offset + len(results),
            'page': page_stem,
            'row_in_page': strip_idx,
            'present': density > density_threshold,
            'density': round(float(density), 5),
        })

    if debug:
        _save_debug_image(image, row_strips, seance_x1, seance_x2, results, debug_root, page_stem)

    return results

def _binarize(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary_inv = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    return cv2.bitwise_not(binary_inv)

def _find_row_boundaries(binary: np.ndarray, min_span_ratio: float = 0.20) -> List[int]:
    h, w = binary.shape
    inv = cv2.bitwise_not(binary)
    kernel_w = max(10, int(w * min_span_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)
    row_sums = np.sum(h_lines > 0, axis=1)
    threshold = w * min_span_ratio * 0.5
    detected = np.where(row_sums > threshold)[0]
    if len(detected) == 0:
        return [0, h]
    return _group_consecutive(detected, gap=6)

def _find_col_boundaries(binary: np.ndarray, min_span_ratio: float = 0.15) -> List[int]:
    h, w = binary.shape
    inv = cv2.bitwise_not(binary)
    kernel_h = max(10, int(h * min_span_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)
    col_sums = np.sum(v_lines > 0, axis=0)
    threshold = h * min_span_ratio * 0.5
    detected = np.where(col_sums > threshold)[0]
    if len(detected) == 0:
        return [0, w]
    return _group_consecutive(detected, gap=6)

def _group_consecutive(positions: np.ndarray, gap: int = 5) -> List[int]:
    if len(positions) == 0:
        return []
    groups = []
    current = [int(positions[0])]
    for pos in positions[1:]:
        if int(pos) - current[-1] <= gap:
            current.append(int(pos))
        else:
            groups.append(int(np.mean(current)))
            current = [int(pos)]
    groups.append(int(np.mean(current)))
    return groups

def _get_target_col_bounds(col_bounds: List[int], active_seance: int, width: int) -> Tuple[int, int]:
    if len(col_bounds) < 3:
        W = width
        COL_NOM_END = 0.23
        COL_SEANCE_W = 0.077
        x1 = int(W * (COL_NOM_END + (active_seance - 1) * COL_SEANCE_W))
        x2 = int(W * (COL_NOM_END + active_seance * COL_SEANCE_W))
        return x1, x2

    max_w = 0
    max_idx = -1
    for i in range(len(col_bounds) - 1):
        w = col_bounds[i+1] - col_bounds[i]
        if w > max_w:
            max_w = w
            max_idx = i

    target_idx = max_idx + active_seance
    if target_idx + 1 < len(col_bounds):
        return col_bounds[target_idx], col_bounds[target_idx + 1]
    else:
        if target_idx < len(col_bounds):
            x1 = col_bounds[target_idx]
        else:
            x1 = col_bounds[-1]
        return x1, x1 + int(width * 0.077)

def _get_robust_row_strips(row_bounds: List[int]) -> List[Tuple[int, int]]:
    heights = []
    for i in range(len(row_bounds) - 1):
        h = row_bounds[i+1] - row_bounds[i]
        heights.append(h)
    
    if not heights:
        return []
        
    plausible = [h for h in heights if h < 150]
    if not plausible:
        return []
        
    median_h = np.median(plausible)
    strips = []
    for i in range(len(row_bounds) - 1):
        y1 = row_bounds[i]
        y2 = row_bounds[i+1]
        h = y2 - y1
        if median_h * 0.65 <= h <= median_h * 1.5:
            strips.append((y1, y2))
            
    return strips

def _save_debug_image(image: np.ndarray, row_strips: List[Tuple[int, int]], seance_x1: int, seance_x2: int,
                      results: List[Dict], debug_root: str, page_stem: str) -> None:
    os.makedirs(debug_root, exist_ok=True)
    dbg = image.copy()
    h_img = dbg.shape[0]

    cv2.line(dbg, (seance_x1, 0), (seance_x1, h_img), (0, 165, 255), 2)
    cv2.line(dbg, (seance_x2, 0), (seance_x2, h_img), (0, 165, 255), 2)

    for i, (y1, y2) in enumerate(row_strips):
        cv2.line(dbg, (0, y1), (dbg.shape[1], y1), (255, 0, 0), 1)
        if i >= len(results):
            continue
        color = (0, 200, 0) if results[i]['present'] else (0, 0, 200)
        overlay = dbg.copy()
        cv2.rectangle(overlay, (seance_x1, y1), (seance_x2, y2), color, cv2.FILLED)
        cv2.addWeighted(overlay, 0.35, dbg, 0.65, 0, dbg)
        cv2.rectangle(dbg, (seance_x1, y1), (seance_x2, y2), color, 1)

        label = f"{results[i]['density']:.3f}"
        cv2.putText(dbg, label, (seance_x1 + 2, y1 + (y2 - y1) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

    out_path = os.path.join(debug_root, f"{page_stem}_debug.jpg")
    cv2.imwrite(out_path, dbg)
    print(f"[checkbox]   Debug image saved -> {out_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/checkbox_detection.py <doc_id> [seance] [threshold] [debug]")
        sys.exit(1)
    _doc_id = sys.argv[1]
    _seance = int(sys.argv[2]) if len(sys.argv) > 2 else None
    _threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.04
    _debug = (sys.argv[4].lower() == "debug") if len(sys.argv) > 4 else False
    _results = detect_absences(doc_id=_doc_id, active_seance=_seance, density_threshold=_threshold, debug=_debug)
    print("\nRow   Page             PgRow  Status    Density")
    print("-" * 52)
    for r in _results:
        status = "PRESENT" if r['present'] else "absent "
        print(f"{r['global_row']:>4}  {r['page']:<14}  {r['row_in_page']:>5}  {status}  {r['density']:>8.5f}")
    present = sum(1 for r in _results if r['present'])
    absent = len(_results) - present
    print(f"\nTotal: {len(_results)}   Present: {present}   Absent: {absent}")