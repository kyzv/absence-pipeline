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

def is_blank(cell_img, dark_pixel_threshold=50):
    """Return True if the cell is virtually empty."""
    if cell_img.size == 0:
        return True
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    dark_pixels = cv2.countNonZero(binary)
    return dark_pixels < dark_pixel_threshold

def is_dense_signature(cell_img):
    """Return True if cell has very high density of dark pixels (likely a signature)."""
    if cell_img.size == 0:
        return False
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    dark_pixels = cv2.countNonZero(binary)
    # If more than 15% of the cell is dark pixels, it's likely a dense signature
    return (dark_pixels / cell_img.size) > 0.15

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

def process_document(doc_id, debug=False):
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
    mapped_csv_file = map_filiere_to_csv(filiere_name)
    
    if not mapped_csv_file:
        print(f"WARNING: Could not map filiere '{filiere_name}' to any CSV file in config/groups.")
        student_df = pd.DataFrame(columns=["n_apo", "nom", "prenom"])
    else:
        csv_path = os.path.join("config", "groups", mapped_csv_file)
        if os.path.exists(csv_path):
            student_df = pd.read_csv(csv_path)
            doc_data["student_list_csv"] = mapped_csv_file
            print(f"Mapped filiere '{filiere_name}' to CSV: {mapped_csv_file}")
        else:
            print(f"ERROR: Mapped CSV {csv_path} does not exist.")
            student_df = pd.DataFrame(columns=["n_apo", "nom", "prenom"])

    row_files = natural_sorted(glob.glob(f"{rows_dir}/row_*.jpg"))
    if not row_files:
        print(f"ERROR: No row images found in {rows_dir}")
        return

    print(f"[{doc_id}] Processing {len(row_files)} physical student rows...")
    
    # Check how many seances are expected based on the JSON
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
            if cell.shape[1] > 0 and cell.shape[0] > 0 and not is_blank(cell, 20):
                text, _ = extract_ocr_with_confidence(cell)
                cleaned = clean_ocr_text(text)
                if cleaned:
                    student_parts.append(cleaned)
                    
        student_name_ocr = " ".join(student_parts)
        
        # Stop condition
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

    # 2. Match physical rows to official DB list
    # Initialize the output array exactly sized to the DB
    absences_output = [None] * len(student_df)
    used_physical = set()
    
    for idx, row in student_df.iterrows():
        official_n_apo = str(row.get("n_apo", ""))
        official_nom = str(row.get("nom", ""))
        official_prenom = str(row.get("prenom", ""))
        official_full = f"{official_n_apo} {official_nom} {official_prenom}".strip()
        
        best_match_idx = -1
        best_conf = 0.0
        
        # Greedy search for the best physical row
        for p_idx, p_data in enumerate(physical_data):
            if p_idx in used_physical:
                continue
            
            # Use fuzz.token_set_ratio which is good for names in different orders
            conf = fuzz.token_set_ratio(official_full.lower(), p_data["ocr_raw"].lower()) / 100.0
            if conf > best_conf:
                best_conf = conf
                best_match_idx = p_idx
                
        # If we found a good match
        if best_match_idx != -1 and best_conf > 0.4:
            used_physical.add(best_match_idx)
            p_data = physical_data[best_match_idx]
            
            # Build sessions data
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
                
                is_present = True
                detected_by = "unknown"
                ocr_conf = 0.0
                
                if cell.shape[1] > 5 and cell.shape[0] > 5:
                    if debug:
                        debug_cell_path = os.path.join(debug_dir, f"row{p_data['row_idx']}_seance{seance_idx}.jpg")
                        cv2.imwrite(debug_cell_path, cell)
                        
                    if is_blank(cell, dark_pixel_threshold=30):
                        is_present = False
                        detected_by = "empty_cell"
                        ocr_conf = 100.0
                    elif is_dense_signature(cell):
                        is_present = True
                        detected_by = "signature_density"
                        ocr_conf = 100.0
                    else:
                        text, conf = extract_ocr_with_confidence(cell)
                        text_lower = text.lower()
                        if text_lower in ['a', 'abs', 'absent']:
                            is_present = False
                            detected_by = "text_abs"
                            ocr_conf = conf
                        elif text_lower in ['p', 'pr', 'present']:
                            is_present = True
                            detected_by = "text_present"
                            ocr_conf = conf
                        else:
                            is_present = True
                            detected_by = "noise_or_signature"
                            ocr_conf = conf

                sessions_obj[f"seance{seance_idx}"] = {
                    "is_present": is_present,
                    "detected_by": detected_by,
                    "ocr_confidence": ocr_conf
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
            # Not found on physical sheet
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
    
    # Save single unified JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(doc_data, f, indent=4, ensure_ascii=False)
        
    print(f"  [OK] Matched {len(used_physical)} rows out of {len(student_df)} official students for {doc_id}")
    print(f"  Written to {json_path}")
    
    # Save pipeline-compatible metadata.json and absences.json
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
    args = parser.parse_args()
    process_document(args.doc_id, args.debug)
