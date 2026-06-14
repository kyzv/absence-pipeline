import os
import cv2
import numpy as np
import glob
import argparse
import re

def natural_sorted(files):
    def key(f):
        m = re.search(r'(\d+)', f)
        return int(m.group(1)) if m else 0
    return sorted(files, key=key)

def get_vertical_lines(img):
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

def process_document(doc_id):
    rows_dir = f"data/rows/{doc_id}"
    cells_dir = f"data/cells/{doc_id}"
    
    if not os.path.exists(rows_dir):
        print(f"ERROR: No rows found for {doc_id} in {rows_dir}")
        return
        
    row_files = natural_sorted(glob.glob(f"{rows_dir}/row_*.jpg"))
    if not row_files:
        print(f"ERROR: No row images found in {rows_dir}")
        return
        
    os.makedirs(cells_dir, exist_ok=True)
    print(f"Slicing cells for {len(row_files)} rows in {doc_id}...")
    
    for row_path in row_files:
        filename = os.path.basename(row_path)
        row_name = os.path.splitext(filename)[0]
        
        img = cv2.imread(row_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        v_lines = get_vertical_lines(img)
        
        # We need at least 4 vertical lines (N_apo, Nom, Prenom, Seance1...)
        if len(v_lines) < 4:
            continue
            
        row_cell_dir = os.path.join(cells_dir, row_name)
        os.makedirs(row_cell_dir, exist_ok=True)
        
        # Slicing seances (starts at index 3: N_apo=0, Nom=1, Prenom=2)
        seance_idx = 1
        for i in range(3, len(v_lines) - 1):
            x1, x2 = v_lines[i], v_lines[i+1]
            # Crop exactly the cell minus a 2px padding to remove grid borders
            cell = img[2:max(3, h-2), x1+2:max(x1+3, x2-2)]
            
            if cell.shape[1] > 5 and cell.shape[0] > 5:
                cv2.imwrite(os.path.join(row_cell_dir, f"seance_{seance_idx}.jpg"), cell)
                
            seance_idx += 1
            if seance_idx > 10:  # Max 10 seances expected
                break
                
    print(f"Done. Cell images saved to {cells_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Slice row images into individual cell images.")
    parser.add_argument('doc_id', help="Document ID to process (e.g., doc_1)")
    args = parser.parse_args()
    process_document(args.doc_id)
