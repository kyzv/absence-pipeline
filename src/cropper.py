# src/cropper.py
import os
import cv2
import numpy as np
from typing import Dict, Optional, List


def crop_document(doc_id: str,
                  preprocessed_root: str = "data/preprocessed",
                  cropped_root: str = "data/cropped") -> List[Dict]:
    prep_dir = os.path.join(preprocessed_root, doc_id)
    if not os.path.isdir(prep_dir):
        raise FileNotFoundError(f"Preprocessed folder not found: {prep_dir}")

    colour_files = _natural_sorted([
        f for f in os.listdir(prep_dir)
        if f.lower().endswith('.jpeg')
    ])

    results = []
    for idx, cf in enumerate(colour_files):
        base = os.path.splitext(cf)[0]
        binary_file = base + "_binary.jpg"
        colour_path = os.path.join(prep_dir, cf)
        binary_path = os.path.join(prep_dir, binary_file)

        if not os.path.exists(binary_path):
            raise FileNotFoundError(f"Missing binary file for {cf}: {binary_path}")

        is_first = (idx == 0)
        res = _crop_one(binary_path, colour_path, doc_id, is_first, cropped_root)
        results.append(res)

    return results


def _crop_one(binary_path: str, colour_path: str,
              doc_id: str, is_first_page: bool,
              output_root: str) -> Dict:
    # Load binary as BGR (3 channels) so _detect_grey_row works.
    binary = cv2.imread(binary_path)          # <-- this was the fix
    colour = cv2.imread(colour_path)
    if binary is None or colour is None:
        raise FileNotFoundError(f"Cannot read: {binary_path} or {colour_path}")

    if is_first_page:
        split_y = _detect_grey_row(binary)
        if split_y is None:
            split_y = int(colour.shape[0] * 0.30)
        split_y = max(10, min(colour.shape[0] - 10, split_y))
    else:
        split_y = 0

    out_dir = os.path.join(output_root, doc_id)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(colour_path))[0]

    if split_y > 0:
        header = colour[:split_y, :]
        table  = colour[split_y:, :]
        header_path = os.path.join(out_dir, f"{base}_header.jpg")
        table_path  = os.path.join(out_dir, f"{base}_table.jpg")
        cv2.imwrite(header_path, header)
        cv2.imwrite(table_path, table)
        return {'has_header': True, 'header_path': header_path, 'table_path': table_path}
    else:
        table_path = os.path.join(out_dir, f"{base}_table.jpg")
        cv2.imwrite(table_path, colour)
        return {'has_header': False, 'header_path': None, 'table_path': table_path}


def _detect_grey_row(image: np.ndarray) -> Optional[int]:
    bgr = image.astype(np.float32)
    min_vals = np.min(bgr, axis=2)
    max_vals = np.max(bgr, axis=2)
    diff = max_vals - min_vals
    mean_bright = np.mean(bgr, axis=2)

    is_grey = (diff < 25) & (mean_bright > 100) & (mean_bright < 200)
    grey_counts = np.sum(is_grey, axis=1)

    width_threshold = image.shape[1] * 0.70
    candidate_rows = np.where(grey_counts > width_threshold)[0]

    if len(candidate_rows) == 0:
        return None

    best_row = candidate_rows[np.argmax(grey_counts[candidate_rows])]

    top = best_row
    while top > 0 and grey_counts[top - 1] > width_threshold:
        top -= 1
    bottom = best_row
    while bottom < image.shape[0] - 1 and grey_counts[bottom + 1] > width_threshold:
        bottom += 1

    return bottom + 1


def _natural_sorted(files: List[str]) -> List[str]:
    import re
    def key(f):
        m = re.search(r'(\d+)', f)
        return int(m.group(1)) if m else 0
    return sorted(files, key=key)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/cropper.py <doc_id>")
        sys.exit(1)
    doc_id = sys.argv[1]
    results = crop_document(doc_id)
    for r in results:
        status = "header+table" if r['has_header'] else "table only"
        print(f"Page: {status}, table: {r['table_path']}")