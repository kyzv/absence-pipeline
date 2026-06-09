import os
import cv2
import csv
import numpy as np
from typing import Dict, List, Tuple
from thefuzz import process, fuzz

def find_vertical_lines(binary: np.ndarray, header_y: int) -> List[int]:
    """Finds exact X coordinates of physical vertical lines in the table."""
    # binary has ink=0, bg=255. Invert it so ink=255.
    binary_inv = cv2.bitwise_not(binary)
    # Only look below header
    table_area = binary_inv[int(header_y):, :]
    
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(table_area.shape[0] * 0.15)))
    v_lines_img = cv2.morphologyEx(table_area, cv2.MORPH_OPEN, v_kernel, iterations=1)
    
    v_sums = np.sum(v_lines_img, axis=0)
    # Threshold for a line is e.g. 10% of the table height being solid black ink
    threshold = 255 * table_area.shape[0] * 0.1
    v_peaks = np.where(v_sums > threshold)[0]
    
    if len(v_peaks) == 0:
        return []
        
    # Group consecutive pixels belonging to the same line
    groups = []
    current = [v_peaks[0]]
    for p in v_peaks[1:]:
        if p - current[-1] <= 15: # Line can be a few pixels wide
            current.append(p)
        else:
            groups.append(int(np.mean(current)))
            current = [p]
    groups.append(int(np.mean(current)))
    return groups

def analyze_table(image: np.ndarray,
                  binary: np.ndarray,
                  ocr_blocks: List[Dict],
                  student_csv_path: str,
                  header_y: int,
                  doc_id: str,
                  debug_dir: str = "data/debug") -> Dict:
    
    if not os.path.exists(student_csv_path):
        raise FileNotFoundError(f"Student CSV not found: {student_csv_path}")
        
    students_db = []
    with open(student_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if not headers:
            return []
            
        nom_col = next((h for h in headers if 'nom' in h.lower() or 'prénom' in h.lower()), headers[-1])
        id_col = next((h for h in headers if 'apo' in h.lower() or 'id' in h.lower() or 'n°' in h.lower()), headers[0])
        
        for row in reader:
            students_db.append({
                'n_apo': row[id_col],
                'nom': row[nom_col]
            })
            
    # 1. Find physical columns
    v_lines = find_vertical_lines(binary, header_y)
    
    # Fallback to OCR bounds if lines not found
    seances_cols = {}
    nom_x_min, nom_x_max = 50, int(image.shape[1] * 0.3)
    
    # Use OCR to label the physical columns
    for block in ocr_blocks:
        text = block['text'].lower()
        x, y = block['center']
        
        if y > header_y:
            continue
            
        if 'nom' in text or 'prénom' in text or 'prenom' in text:
            # Find closest vertical lines enclosing this text
            if len(v_lines) >= 2:
                left_lines = [l for l in v_lines if l < x]
                right_lines = [l for l in v_lines if l > x]
                if left_lines: nom_x_min = max(left_lines)
                if right_lines: nom_x_max = min(right_lines)
                
        if 'séance' in text or 'seance' in text:
            import re
            m = re.search(r'seance\s*(\d+)|séance\s*(\d+)', text)
            if m:
                s_num = m.group(1) or m.group(2)
                # Enclosing lines
                if len(v_lines) >= 2:
                    left_lines = [l for l in v_lines if l < x]
                    right_lines = [l for l in v_lines if l > x]
                    if left_lines and right_lines:
                        seances_cols[s_num] = {
                            'x_min': max(left_lines),
                            'x_max': min(right_lines),
                            'has_data': False
                        }

    # Group OCR blocks into rows
    nom_blocks = []
    for block in ocr_blocks:
        x, y = block['center']
        if y > header_y and nom_x_min <= x <= nom_x_max:
            if block['text'].lower() not in ['nom', 'prénom', 'prenom', 'absent', 'présent'] and len(block['text']) > 2:
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
    
    for r_idx, row_blocks in enumerate(rows):
        row_y = sum(b['center'][1] for b in row_blocks) / len(row_blocks)
        raw_name = " ".join([b['text'] for b in sorted(row_blocks, key=lambda b: b['center'][0])])
        
        # Levenshtein Matching
        best_match, score = process.extractOne(raw_name, [s['nom'] for s in students_db], scorer=fuzz.token_set_ratio)
        if score < 50:
            continue
            
        matched_student = next(s for s in students_db if s['nom'] == best_match)
        
        y_min = int(row_y - 15)
        y_max = int(row_y + 15)
        
        student_absences = []
        
        for s_num, bounds in seances_cols.items():
            x_min = int(bounds['x_min'])
            x_max = int(bounds['x_max'])
            
            # 1. OCR fallback check
            cell_text = ""
            for b in ocr_blocks:
                bx, by = b['center']
                if x_min <= bx <= x_max and y_min <= by <= y_max:
                    cell_text += b['text'].lower() + " "
                    
            if 'abs' in cell_text or 'a' in cell_text.split() or 'absent' in cell_text:
                status = "Absent"
                bounds['has_data'] = True
            else:
                # 2. Pixel density
                cell_img = binary[max(0, y_min):min(binary.shape[0], y_max), 
                                  max(0, x_min):min(binary.shape[1], x_max)]
                
                if cell_img.size == 0:
                    density = 0
                else:
                    # In our binary, ink is 0, background is 255
                    black_pixels = np.sum(cell_img == 0)
                    density = black_pixels / cell_img.size
                
                if density > 0.015: # Tuned threshold for signatures
                    status = "Present"
                    bounds['has_data'] = True
                else:
                    status = "Absent"
                    
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
        
    # Filter active sessions
    active_sessions = []
    for s_num in seances_cols.keys():
        is_active = False
        for r in results:
            for s in r['sessions']:
                if s['seance'] == s_num and (s['status'] == 'Present' or seances_cols[s_num]['has_data']):
                    is_active = True
                    break
            if is_active: break
        if is_active:
            active_sessions.append(s_num)
            
    final_results = []
    for r in results:
        active_s = [s for s in r['sessions'] if s['seance'] in active_sessions]
        r['sessions'] = active_s
        final_results.append(r)

    # Debug image
    dbg = image.copy()
    cv2.line(dbg, (0, int(header_y)), (dbg.shape[1], int(header_y)), (0, 0, 255), 2)
    
    # Draw physical lines detected
    for vl in v_lines:
        cv2.line(dbg, (vl, 0), (vl, dbg.shape[0]), (255, 0, 0), 1)
        
    for r in final_results:
        for s in r['sessions']:
            x1, y1, x2, y2 = s['box']
            color = (0, 255, 0) if s['status'] == "Present" else (0, 0, 255)
            cv2.rectangle(dbg, (x1, y1), (x2, y2), color, 2)
            cv2.putText(dbg, s['status'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
    os.makedirs(os.path.join(debug_dir, doc_id), exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, doc_id, "table_debug.jpg"), dbg)

    return final_results
