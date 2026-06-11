import os
import cv2
import glob
import numpy as np
import argparse
import re
import json
import pytesseract
import pandas as pd
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
    
    # Use a kernel almost as tall as the row to catch vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, img.shape[0] - 10)))
    v_morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    v_proj = np.sum(v_morphed, axis=0)
    
    v_lines = []
    in_line = False
    start_x = 0
    threshold = 255 * (img.shape[0] - 15) # threshold to consider a column as a line
    
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

def clean_ocr_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).strip()

def process_document(doc_id, debug=False):
    rows_dir = f"data/rows/{doc_id}"
    json_path = f"data/output/{doc_id}.json"
    
    if not os.path.exists(rows_dir):
        print(f"ERROR: Row directory not found: {rows_dir}")
        return
        
    if not os.path.exists(json_path):
        print(f"ERROR: JSON output not found: {json_path}. Run ocr_header.py first.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        doc_data = json.load(f)

    # Load filiere and map to CSV
    filiere_name = doc_data.get("filiere", {}).get("value", "")
    mapped_csv_file = map_filiere_to_csv(filiere_name)
    
    if not mapped_csv_file:
        print(f"WARNING: Could not map filiere '{filiere_name}' to any CSV file in config/groups.")
        # Fallback empty dataframe
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

    print(f"[{doc_id}] Processing {len(row_files)} student rows...")
    
    absences = []
    
    # Check how many seances are expected based on the JSON
    seances = doc_data.get("seances", {})
    expected_seance_count = len(seances)
    
    for row_path in row_files:
        filename = os.path.basename(row_path)
        img = cv2.imread(row_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # We need vertical lines to define columns
        v_lines = get_vertical_lines(img)
        
        # If we didn't find enough vertical lines, we might need a fallback or to skip
        # Expecting at least: left edge, ID edge, Name edge, Surname edge, ... seance edges, right edge
        # Minimum 4 lines for just ID, Nom, Prenom
        if len(v_lines) < 4:
            if debug:
                print(f"  Warning: Not enough vertical lines found in {filename} ({len(v_lines)}). Skipping.")
            continue
            
        # Ensure 0 and W are in the lines list if they are missing
        if v_lines[0] > 10:
            v_lines.insert(0, 0)
        if w - v_lines[-1] > 10:
            v_lines.append(w)
            
        # Extract the first 3 columns (assuming ID, Nom, Prénom)
        student_parts = []
        for i in range(min(3, len(v_lines) - 1)):
            x1, x2 = v_lines[i], v_lines[i+1]
            cell = img[2:h-2, x1+2:x2-2] # crop inwards slightly
            if cell.shape[1] > 0 and cell.shape[0] > 0 and not is_blank(cell, 20):
                text = pytesseract.image_to_string(cell, config='--psm 7').strip()
                cleaned = clean_ocr_text(text)
                if cleaned:
                    student_parts.append(cleaned)
                    
        student_name_ocr = " ".join(student_parts)
        
        # Stop condition: if this is the EMARGEMENT row for teachers
        from rapidfuzz import fuzz
        # Check if the OCR text itself strongly matches "EMARGEMENT" or "signature"
        if "emargement" in student_name_ocr.lower() or fuzz.partial_ratio("emargement", student_name_ocr.lower()) > 80:
            print(f"  Stopping at {filename}: Found EMARGEMENT row.")
            break
            
        # If student name is empty, it might be a blank row or noise
        if not student_name_ocr:
            continue
            
        # Use Levenshtein distance to match with official student list
        matched_student, confidence = match_student(student_name_ocr, student_df)
        
        if matched_student and confidence > 0.4:
            row_data = {
                "n_apo": matched_student["n_apo"],
                "nom": matched_student["nom"],
                "prenom": matched_student["prenom"],
                "ocr_raw": student_name_ocr,
                "confidence": confidence,
            }
        else:
            row_data = {
                "n_apo": "",
                "nom": student_name_ocr, # Fallback to raw OCR text
                "prenom": "",
                "ocr_raw": student_name_ocr,
                "confidence": confidence,
            }

        # Extract remaining columns as seances
        # We map col[3] to seance1, col[4] to seance2, etc.
        seance_idx = 1
        row_sessions = []
        for i in range(3, len(v_lines) - 1):
            if seance_idx > expected_seance_count:
                break # Don't parse more seances than defined in header
                
            x1, x2 = v_lines[i], v_lines[i+1]
            cell = img[2:h-2, x1+2:x2-2]
            
            # The absence logic
            status = "Present"
            
            if cell.shape[1] > 5 and cell.shape[0] > 5: # ensure it's not a tiny sliver
                if is_blank(cell, dark_pixel_threshold=30):
                    status = "Absent"
                else:
                    text = pytesseract.image_to_string(cell, config='--psm 7').strip().lower()
                    if text in ['a', 'abs', 'absent']:
                        status = "Absent"
                    # Everything else is Present (e.g. signatures, checkmarks, 'P')
                    
            row_data[f"seance{seance_idx}"] = status
            row_sessions.append({
                "seance": str(seance_idx),
                "status": status
            })
            seance_idx += 1
            
        row_data["sessions"] = row_sessions
        absences.append(row_data)
        
    # Remove sessions key from absences array before saving to doc_data to keep compatibility with old format
    # But wait, pipeline.py needs sessions key in absences.json, so we will generate both!
    doc_data["absences"] = absences
    
    # 1. Save single unified JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(doc_data, f, indent=4, ensure_ascii=False)
        
    print(f"  [OK] Processed {len(absences)} student rows for {doc_id}")
    print(f"  Written to {json_path}")
    
    # 2. Save pipeline-compatible metadata.json and absences.json
    out_dir = os.path.join("data", "output", doc_id)
    os.makedirs(out_dir, exist_ok=True)
    
    metadata_compatible = {k: v for k, v in doc_data.items() if k != "absences"}
    with open(os.path.join(out_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata_compatible, f, indent=4, ensure_ascii=False)
        
    with open(os.path.join(out_dir, "absences.json"), 'w', encoding='utf-8') as f:
        json.dump(absences, f, indent=4, ensure_ascii=False)
        
    print(f"  Written compatible splits to {out_dir}/metadata.json and absences.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract student absences from row images.")
    parser.add_argument('doc_id', help="Document ID to process (e.g., doc_8)")
    parser.add_argument('--debug', action='store_true', help="Enable debug output")
    args = parser.parse_args()
    process_document(args.doc_id, args.debug)
