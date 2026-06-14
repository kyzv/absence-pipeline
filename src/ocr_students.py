import os
import cv2
import glob
import numpy as np
import argparse
import re
import json
import pytesseract
from pytesseract import Output
import pandas as pd
from rapidfuzz import fuzz
from name_matcher import map_filiere_to_csv, match_student

def natural_sorted(files):
    def key(f):
        m = re.search(r'(\d+)', f)
        return int(m.group(1)) if m else 0
    return sorted(files, key=key)

def get_vertical_lines(img):
    """Find the x-coordinates of vertical grid lines."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, img.shape[0] - 10)))
    v_morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    v_proj = np.sum(v_morphed, axis=0)
    
    v_lines = []
    in_line = False
    start_x = 0
    threshold = 255 * (img.shape[0] - 15)
    
    for x, val in enumerate(v_proj):
        if val > threshold:
            if not in_line: 
                start_x = x
                in_line = True
        else:
            if in_line: 
                v_lines.append(int((start_x + x - 1) / 2))
                in_line = False
    if in_line: 
        v_lines.append(int((start_x + len(v_proj) - 1) / 2))
        
    return v_lines

def pixel_density(cell_img, threshold=200):
    """Fraction of pixels darker than threshold (fixed)."""
    if cell_img.size == 0:
        return 0.0
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return cv2.countNonZero(binary) / (cell_img.shape[0] * cell_img.shape[1])

def clean_ocr_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).strip()

def extract_ocr_with_confidence(cell_img):
    """Extract text and confidence score from a cell using image_to_data."""
    d = pytesseract.image_to_data(cell_img, config='--psm 7', output_type=Output.DICT)
    
    texts = []
    confs = []
    for i in range(len(d['text'])):
        t = d['text'][i].strip()
        c = float(d['conf'][i])
        if t and c >= 0:
            texts.append(t)
            confs.append(c)
            
    if not texts:
        return "", 0.0
        
    full_text = " ".join(texts)
    avg_conf = sum(confs) / len(confs)
    return full_text, avg_conf

# ---- Status-word patterns to detect from OCR ----
_ABSENCE_WORDS = {'a', 'ab', 'abs', 'abst', 'absent', 'as', 'b', 'aa'}
_PRESENT_WORDS = {'p', 'pr', 'pre', 'pres', 'present'}

def _ocr_for_status(gray_upscaled):
    """
    Try to read status markers ('A', 'ABS', 'P', 'PR') from a preprocessed gray image.
    Returns (status: str ['ABSENT', 'PRESENT', None], confidence: float).
    """
    # PSM 10 = single character  (best for lone 'A' or 'P')
    cfg10 = r'--psm 10 --oem 1 -c tessedit_char_whitelist=AaBbSsPpRrEeNnTt'
    # PSM 7  = single line       (best for 'ABS', 'Abs', 'PR')
    cfg7  = r'--psm 7  --oem 1 -c tessedit_char_whitelist=AaBbSsPpRrEeNnTt'

    results = []
    for cfg in (cfg10, cfg7):
        try:
            d = pytesseract.image_to_data(gray_upscaled, config=cfg, output_type=Output.DICT)
            words = [(d['text'][i].strip().lower(), float(d['conf'][i]))
                     for i in range(len(d['text']))
                     if d['text'][i].strip() and float(d['conf'][i]) >= 0]
            if words:
                text = " ".join(w for w, _ in words)
                conf = sum(c for _, c in words) / len(words)
                results.append((text, conf))
        except Exception:
            pass

    for text, conf in results:
        for token in text.split():
            if token in _ABSENCE_WORDS:
                return 'ABSENT', max(conf, 55.0)
            if token in _PRESENT_WORDS:
                return 'PRESENT', max(conf, 55.0)
    return None, 0.0

def classify_seance_cell(cell_img):
    """
    Classify a seance cell as present or absent.

    Returns (is_present: bool, confidence: float [0-100])

    Decision logic:
      1. Truly blank  (density < 0.012)  → ABSENT  95%
      2. OCR detects 'A'/'ABS'           → ABSENT  max(ocr_conf, 60)
      3. OCR detects 'P'/'PR'            → PRESENT max(ocr_conf, 60)
      4. Contour/Density                 → PRESENT (if max contour > 150 or density > 0.08)
      5. Ambiguous (mid density, no OCR) → PRESENT ~45% (needs admin review)
    """
    if cell_img is None or cell_img.size == 0:
        return False, 50.0
    
    h, w = cell_img.shape[:2]
    if h < 4 or w < 4:
        return False, 50.0

    # --- Step 1: pixel density (fixed threshold 200) ---
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    dark_pixels = cv2.countNonZero(binary)
    density = dark_pixels / (h * w)

    BLANK_DENSITY = 0.012

    if density < BLANK_DENSITY:
        return False, 95.0   # clearly empty → absent

    # --- Step 2: prep image for OCR ---
    # 3x upscale with bicubic (Tesseract performs better on larger images)
    up = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    # CLAHE: normalise contrast (helps with red/faded ink)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    up = clahe.apply(up)
    # Otsu binarise (invert so text is white on black for Tesseract)
    _, thresh_up = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Slight dilation to reconnect broken strokes
    kern = np.ones((2, 2), np.uint8)
    thresh_up = cv2.dilate(thresh_up, kern, iterations=1)

    status, ocr_conf = _ocr_for_status(thresh_up)
    if status == 'ABSENT':
        return False, round(ocr_conf, 1)
    elif status == 'PRESENT':
        return True, round(ocr_conf, 1)

    # --- Step 3: Explicit wet signature detection ---
    # We use OpenCV contours to find large sprawling structures typical of signatures
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour_area = max([cv2.contourArea(c) for c in contours]) if contours else 0
    
    SIGNATURE_DENSITY_THRESH = 0.08
    SIGNATURE_AREA_THRESH = 150.0  # Pixels squared. A single "P" or "A" is usually smaller

    if density > SIGNATURE_DENSITY_THRESH or max_contour_area > SIGNATURE_AREA_THRESH:
        # It's a signature!
        conf = min(85.0, round(density * 400 + max_contour_area * 0.1, 1))
        return True, conf

    # --- Step 4: ambiguous zone ---
    # Small mark but OCR found nothing; could be faint signature or unread 'A'
    # Default PRESENT with low confidence so admin can review
    return True, round(density * 300, 1)

def process_document(doc_id, debug=False, csv_override=None):
    rows_dir = f"data/rows/{doc_id}"
    json_path = f"data/output/{doc_id}.json"
    debug_dir = f"debug/{doc_id}"
    
    if debug:
        os.makedirs(debug_dir, exist_ok=True)
    
    if not os.path.exists(rows_dir):
        print(f"ERROR: Row directory not found: {rows_dir}")
        return
        
    if not os.path.exists(json_path):
        alt_path = f"data/output/{doc_id}/metadata.json"
        if os.path.exists(alt_path):
            json_path = alt_path
        else:
            print(f"ERROR: JSON output not found: {json_path}. Run ocr_header.py first.")
            return

    with open(json_path, 'r', encoding='utf-8') as f:
        doc_data = json.load(f)

    # Load filiere and map to CSV
    filiere_name = doc_data.get("filiere", {}).get("value", "")
    
    if csv_override:
        mapped_csv_file = csv_override
        print(f"Using explicitly provided CSV: {mapped_csv_file}")
    else:
        mapped_csv_file = map_filiere_to_csv(filiere_name)
        if mapped_csv_file:
            print(f"Mapped filiere '{filiere_name}' to CSV: {mapped_csv_file}")
    
    if not mapped_csv_file:
        print(f"WARNING: Could not map filiere '{filiere_name}' to any CSV file in config/groups.")
        student_df = pd.DataFrame(columns=["n_apo", "nom", "prenom"])
    else:
        csv_path = os.path.join("config", "groups", mapped_csv_file)
        if os.path.exists(csv_path):
            student_df = pd.read_csv(csv_path)
            student_df.columns = student_df.columns.str.strip().str.lower()
            doc_data["student_list_csv"] = mapped_csv_file
        else:
            print(f"ERROR: Mapped CSV {csv_path} does not exist.")
            student_df = pd.DataFrame(columns=["n_apo", "nom", "prenom"])

    row_files = natural_sorted(glob.glob(f"{rows_dir}/row_*.jpg"))
    if not row_files:
        print(f"ERROR: No row images found in {rows_dir}")
        return

    print(f"[{doc_id}] Processing {len(row_files)} physical student rows...")
    
    seances = doc_data.get("seances", {})
    expected_seance_count = len(seances)
    
    physical_data = []
    
    # 1. Parse all physical rows
    for row_idx, row_path in enumerate(row_files):
        filename = os.path.basename(row_path)
        img = cv2.imread(row_path)
        if img is None: continue
            
        h, w = img.shape[:2]
        v_lines = get_vertical_lines(img)
        
        if len(v_lines) < 4:
            continue
            
        if v_lines[0] > 10: v_lines.insert(0, 0)
        if w - v_lines[-1] > 10: v_lines.append(w)
            
        student_parts = []
        for i in range(min(3, len(v_lines) - 1)):
            x1, x2 = v_lines[i], v_lines[i+1]
            cell = img[2:h-2, x1+2:x2-2]
            if cell.shape[1] > 5 and cell.shape[0] > 5:
                text, _ = extract_ocr_with_confidence(cell)
                cleaned = clean_ocr_text(text)
                if cleaned:
                    student_parts.append(cleaned)
                    
        student_name_ocr = " ".join(student_parts)
        
        # Stop condition: only if "EMARGEMENT" is the name text in the row
        if "emargement" in student_name_ocr.lower() or fuzz.partial_ratio("emargement", student_name_ocr.lower()) > 80:
            print(f"  Stopping at {filename}: Found EMARGEMENT row.")
            break
            
        if not student_name_ocr:
            continue
            
        physical_data.append({
            "img": img,
            "filename": filename,
            "v_lines": v_lines,
            "ocr_raw": student_name_ocr,
            "h": h,
            "row_idx": row_idx + 1
        })

    # 2. Match physical rows to official DB list (DB order enforced)
    absences_output = [None] * len(student_df)
    used_physical = set()
    
    for idx, row in student_df.iterrows():
        official_n_apo = str(row.get("n_apo", "")).strip()
        official_nom = str(row.get("nom", "")).strip()
        official_prenom = str(row.get("prenom", "")).strip()
        
        official_alpha = f"{official_nom} {official_prenom}".strip().lower()
        
        best_match_idx = -1
        best_conf = 0.0
        
        for p_idx, p_data in enumerate(physical_data):
            if p_idx in used_physical:
                continue
                
            ocr_raw = p_data["ocr_raw"].lower()
            ocr_digits = "".join(c for c in ocr_raw if c.isdigit())
            ocr_alpha = " ".join(w for w in ocr_raw.split() if not w.isdigit())
            
            id_score = 0.0
            if ocr_digits and official_n_apo:
                # If they have digits, check similarity
                from rapidfuzz import distance
                lev = distance.Levenshtein.distance(ocr_digits, official_n_apo)
                m_len = max(len(ocr_digits), len(official_n_apo))
                id_score = (1.0 - lev / m_len) * 100.0 if m_len > 0 else 0.0
                
            name_score = fuzz.token_sort_ratio(official_alpha, ocr_alpha)
            
            if id_score >= 80:
                conf = (0.7 * id_score + 0.3 * name_score) / 100.0
            else:
                conf = name_score / 100.0
                
            if conf > best_conf:
                best_conf = conf
                best_match_idx = p_idx
                
        if best_match_idx != -1 and best_conf > 0.4:
            used_physical.add(best_match_idx)
            p_data = physical_data[best_match_idx]
            
            sessions_obj = {}
            seance_idx = 1
            v_lines = p_data["v_lines"]
            img = p_data["img"]
            h = p_data["h"]
            
            for i in range(3, len(v_lines) - 1):
                if seance_idx > expected_seance_count:
                    break
                    
                x1, x2 = v_lines[i], v_lines[i+1]
                cell = img[2:h-2, x1+2:x2-2]
                
                if cell.shape[1] > 5 and cell.shape[0] > 5:
                    if debug:
                        debug_cell_path = os.path.join(debug_dir, f"row{p_data['row_idx']}_seance{seance_idx}.jpg")
                        cv2.imwrite(debug_cell_path, cell)
                    is_present, ocr_conf = classify_seance_cell(cell)
                else:
                    is_present, ocr_conf = False, 50.0

                sessions_obj[f"seance{seance_idx}"] = {
                    "is_present": is_present,
                    "confidence": round(ocr_conf, 1)
                }
                seance_idx += 1
                
            absences_output[idx] = {
                "row_index": p_data["row_idx"],
                "n_apo": official_n_apo,
                "nom": official_nom,
                "prenom": official_prenom,
                "match_confidence": round(best_conf * 100, 2),
                "ocr_raw_name": p_data["ocr_raw"],
                "sessions": sessions_obj
            }
        else:
            # Not found on physical sheet — include from DB with empty sessions
            absences_output[idx] = {
                "row_index": None,
                "n_apo": official_n_apo,
                "nom": official_nom,
                "prenom": official_prenom,
                "match_confidence": 0.0,
                "ocr_raw_name": "",
                "sessions": {}
            }

    doc_data["absences"] = absences_output
    
    # Save unified JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(doc_data, f, indent=4, ensure_ascii=False)
        
    print(f"  [OK] Matched {len(used_physical)} rows out of {len(student_df)} official students for {doc_id}")
    print(f"  Written to {json_path}")
    
    # Save pipeline-compatible metadata.json + absences.json
    out_dir = os.path.join("data", "output", doc_id)
    os.makedirs(out_dir, exist_ok=True)
    
    metadata_compatible = {k: v for k, v in doc_data.items() if k != "absences"}
    with open(os.path.join(out_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata_compatible, f, indent=4, ensure_ascii=False)
        
    with open(os.path.join(out_dir, "absences.json"), 'w', encoding='utf-8') as f:
        json.dump(absences_output, f, indent=4, ensure_ascii=False)
        
    print(f"  Written compatible splits to {out_dir}/metadata.json and absences.json")
    if debug:
        print(f"  Debug images saved to {debug_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract student absences from row images.")
    parser.add_argument('doc_id', help="Document ID to process (e.g., doc_8)")
    parser.add_argument('--debug', action='store_true', help="Enable debug output")
    parser.add_argument('--csv', help="Override the CSV file mapping", default=None)
    args = parser.parse_args()
    process_document(args.doc_id, args.debug, args.csv)
