# src/checkbox_detection.py
import os, cv2, numpy as np
from typing import List
from paddleocr import PaddleOCR

def detect_absences_one_page(table_image_path: str,
                             seance_number: int,
                             ocr_instance: PaddleOCR = None) -> List[str]:
    img = cv2.imread(table_image_path)
    if img is None:
        raise FileNotFoundError(f"Table image not found: {table_image_path}")
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if ocr_instance is None:
        ocr = PaddleOCR(lang='fr', use_textline_orientation=True)
    else:
        ocr = ocr_instance
    result = ocr.predict(table_image_path)
    if not result or not result[0]:
        return []

    detections = result[0]
    texts = detections.get('rec_texts', [])
    boxes = detections.get('rec_polys', [])

    # --- Find name column width dynamically ---
    name_right_x = 0
    name_rows = []
    for box, text in zip(boxes, texts):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        xc = sum(xs) / len(xs)
        yc = sum(ys) / len(ys)
        # Assume name column is roughly left 50% of the image
        if xc < w * 0.5:
            right_edge = max(xs)
            if right_edge > name_right_x:
                name_right_x = right_edge
            name_rows.append({'y': yc, 'name': text.strip()})

    if not name_rows:
        return []

    # Add a small safety margin
    name_right_x = min(name_right_x + 5, w)

    # Split remaining width into 10 equal séance columns
    seance_width = (w - name_right_x) / 10
    seance_left = int(name_right_x + (seance_number - 1) * seance_width)
    seance_right = int(name_right_x + seance_number * seance_width)

    # --- Group name rows (same as before) ---
    name_rows.sort(key=lambda r: r['y'])
    merged = []
    cur = name_rows[0]
    for r in name_rows[1:]:
        if r['y'] - cur['y'] < 20:
            cur['name'] += ' ' + r['name']
        else:
            merged.append(cur)
            cur = r
    merged.append(cur)

    # --- Decision per row ---
    absent = []
    for row in merged:
        yc = int(row['y'])
        y1 = max(0, yc - 15)
        y2 = min(h, yc + 15)
        cell_gray = gray[y1:y2, seance_left:seance_right]
        if cell_gray.size == 0:
            continue
        _, cell_bin = cv2.threshold(cell_gray, 150, 255, cv2.THRESH_BINARY_INV)
        ink_ratio = np.sum(cell_bin == 255) / cell_bin.size
        # If visibly empty → run tiny OCR to catch a possible "A"
        if ink_ratio < 0.02:
            cell_img = img[y1:y2, seance_left:seance_right]
            cv2.imwrite("temp_cell.jpg", cell_img)
            cell_res = ocr.predict("temp_cell.jpg")
            cell_text = ""
            if cell_res and cell_res[0]:
                ct = cell_res[0].get('rec_texts', [])
                if ct:
                    cell_text = ct[0].strip().upper()
            if not cell_text or any(t in cell_text for t in ('A', 'ABS', 'ABSENT')):
                absent.append(row['name'])
        # else: signed → present, skip

    return absent


def detect_absences_document(doc_id: str,
                             cropped_root: str = "data/cropped",
                             seance_number: int = 1) -> List[str]:
    crop_dir = os.path.join(cropped_root, doc_id)
    table_files = sorted([f for f in os.listdir(crop_dir) if f.endswith("_table.jpg")])
    ocr = PaddleOCR(lang='fr', use_textline_orientation=True)
    all_names = []
    for tf in table_files:
        names = detect_absences_one_page(os.path.join(crop_dir, tf), seance_number, ocr)
        all_names.extend(names)
    return all_names


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python src/checkbox_detection.py <doc_id> <seance_number>")
        sys.exit(1)
    doc_id, seance = sys.argv[1], int(sys.argv[2])
    names = detect_absences_document(doc_id, seance_number=seance)
    print(f"\nAbsent count: {len(names)}")
    for n in names:
        print(n)