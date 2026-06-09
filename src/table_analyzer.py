import os
import cv2
import csv
import numpy as np
from typing import Dict, List, Tuple
from thefuzz import process, fuzz

def analyze_table(image: np.ndarray,
                  binary: np.ndarray,
                  ocr_blocks: List[Dict],
                  student_csv_path: str,
                  header_y: int,
                  doc_id: str,
                  debug_dir: str = "data/debug") -> Dict:
    """
    Analyzes the table area, matches student names, and detects absences.
    """
    if not os.path.exists(student_csv_path):
        raise FileNotFoundError(f"Student CSV not found: {student_csv_path}")
        
    # Load students
    students_db = []
    with open(student_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            students_db.append(row)
            
    # Find columns (Nom, Seance 1..10)
    nom_x_min, nom_x_max = 50, int(image.shape[1] * 0.3) # Defaults
    seances_cols = {}
    
    for block in ocr_blocks:
        text = block['text'].lower()
        y_center = block['center'][1]
        x_center = block['center'][0]
        
        if y_center > header_y:
            continue # We only look at header rows for column bounds
            
        if 'nom' in text or 'prénom' in text or 'prenom' in text:
            xs = [p[0] for p in block['box']]
            nom_x_min = min(xs) - 20
            nom_x_max = max(xs) + 250 # Give enough space for full names
            
        if 'séance' in text or 'seance' in text:
            import re
            m = re.search(r'seance\s*(\d+)|séance\s*(\d+)', text)
            if m:
                s_num = m.group(1) or m.group(2)
                xs = [p[0] for p in block['box']]
                # Usually the column width is around the text width + some margin
                w = max(xs) - min(xs)
                seances_cols[s_num] = {
                    'x_min': min(xs) - 10,
                    'x_max': max(xs) + 10,
                    'has_data': False # Track if this session has happened
                }

    # If we couldn't find seances from header, try to guess based on standard layout
    if not seances_cols:
        W = image.shape[1]
        start_x = int(W * 0.35)
        step_x = int(W * 0.06)
        for i in range(1, 11):
            seances_cols[str(i)] = {
                'x_min': start_x + (i-1)*step_x,
                'x_max': start_x + i*step_x,
                'has_data': False
            }

    # Group OCR blocks into rows
    # Only consider blocks in the Nom column that are below the header
    nom_blocks = []
    for block in ocr_blocks:
        x, y = block['center']
        if y > header_y and nom_x_min <= x <= nom_x_max:
            if block['text'].lower() not in ['nom', 'prénom', 'prenom', 'absent', 'présent']:
                nom_blocks.append(block)
                
    nom_blocks.sort(key=lambda b: b['center'][1])
    
    rows = []
    current_row = []
    for b in nom_blocks:
        if not current_row:
            current_row.append(b)
        else:
            if abs(b['center'][1] - current_row[0]['center'][1]) < 20: # Same row
                current_row.append(b)
            else:
                rows.append(current_row)
                current_row = [b]
    if current_row:
        rows.append(current_row)
        
    results = []
    
    # Analyze each row
    for r_idx, row_blocks in enumerate(rows):
        row_y = sum(b['center'][1] for b in row_blocks) / len(row_blocks)
        raw_name = " ".join([b['text'] for b in sorted(row_blocks, key=lambda b: b['center'][0])])
        
        # Levenshtein Matching
        best_match, score = process.extractOne(raw_name, [s['nom'] for s in students_db], scorer=fuzz.token_set_ratio)
        if score < 50:
            continue # Probably noise
            
        matched_student = next(s for s in students_db if s['nom'] == best_match)
        
        # Check sessions
        y_min = int(row_y - 15)
        y_max = int(row_y + 15)
        
        student_absences = []
        
        for s_num, bounds in seances_cols.items():
            x_min = int(bounds['x_min'])
            x_max = int(bounds['x_max'])
            
            # 1. Check OCR blocks in this cell for "A", "Abs", "Absent"
            cell_text = ""
            for b in ocr_blocks:
                bx, by = b['center']
                if x_min <= bx <= x_max and y_min <= by <= y_max:
                    cell_text += b['text'].lower() + " "
                    
            if 'abs' in cell_text or 'a' in cell_text.split() or 'absent' in cell_text:
                status = "Absent"
                bounds['has_data'] = True
            else:
                # 2. Check pixel density
                cell_img = binary[max(0, y_min):min(binary.shape[0], y_max), 
                                  max(0, x_min):min(binary.shape[1], x_max)]
                
                if cell_img.size == 0:
                    density = 0
                else:
                    black_pixels = np.sum(cell_img == 0) # Ink is 0 in standard binary, wait, our binarize uses bitwise_not so ink is 255.
                    # Wait, let's verify preprocessing.py: cv2.bitwise_not(binary_inv)
                    # ADAPTIVE_THRESH_GAUSSIAN_C with THRESH_BINARY_INV means ink is 255. bitwise_not makes ink 0.
                    # So ink is 0 (black pixels).
                    # Actually, if bitwise_not(binary_inv) -> background is white (255), ink is black (0).
                    # But the previous checkbox_detection.py had: black_pixels = np.sum(cell < 128)
                    density = np.sum(cell_img < 128) / cell_img.size
                
                if density > 0.02: # Threshold
                    status = "Present"
                    bounds['has_data'] = True
                else:
                    status = "Absent" # Empty cell
                    
            student_absences.append({
                "seance": s_num,
                "status": status,
                "density": density if 'density' in locals() else 0.0,
                "box": (x_min, y_min, x_max, y_max)
            })
            
        results.append({
            "n_apo": matched_student['n_apo'],
            "nom": matched_student['nom'],
            "ocr_raw": raw_name,
            "sessions": student_absences,
            "row_y": row_y
        })
        
    # Filter out future/empty sessions
    # If a session column is "Absent" for EVERYONE, it hasn't happened.
    active_sessions = []
    for s_num in seances_cols.keys():
        # Check if anyone is present or has explicit "Abs" text
        is_active = False
        for r in results:
            for s in r['sessions']:
                if s['seance'] == s_num and (s['status'] == 'Present' or seances_cols[s_num]['has_data']):
                    is_active = True
                    break
            if is_active: break
        if is_active:
            active_sessions.append(s_num)
            
    # Clean up results to only include active sessions
    final_results = []
    for r in results:
        active_s = [s for s in r['sessions'] if s['seance'] in active_sessions]
        r['sessions'] = active_s
        final_results.append(r)

    # Debug image
    dbg = image.copy()
    cv2.line(dbg, (0, int(header_y)), (dbg.shape[1], int(header_y)), (0, 0, 255), 2)
    
    for r in final_results:
        for s in r['sessions']:
            x1, y1, x2, y2 = s['box']
            color = (0, 255, 0) if s['status'] == "Present" else (0, 0, 255)
            cv2.rectangle(dbg, (x1, y1), (x2, y2), color, 2)
            cv2.putText(dbg, s['status'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
    os.makedirs(os.path.join(debug_dir, doc_id), exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, doc_id, "table_debug.jpg"), dbg)

    return final_results
