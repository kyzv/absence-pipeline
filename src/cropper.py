"""src/cropper.py"""

import os
import cv2
import glob
import numpy as np
import argparse
import pytesseract
import re

# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def get_horizontal_lines(img):
    """Find the y-coordinates of horizontal grid lines."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Use a large kernel (400px wide) to ignore text and only catch true table lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (400, 1))
    h_morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    h_proj = np.sum(h_morphed, axis=1)
    
    h_lines = []
    in_line = False; start_y = 0
    for y, val in enumerate(h_proj):
        if val > 255 * 400:
            if not in_line: start_y = y; in_line = True
        else:
            if in_line: h_lines.append(int((start_y + y - 1) / 2)); in_line = False
    if in_line: h_lines.append(int((start_y + len(h_proj) - 1) / 2))
    
    return h_lines

def find_anchor_line_y(img, h_lines):
    """
    Verification Loop: Iterates downwards.
    We are looking for the "N° Apo", "Nom", "Prenom" row.
    If OCR fails to find it, we look for the FIRST STUDENT ROW (containing a 7+ digit number).
    If we find a student, we assume the row immediately above it is the header, and crop there.
    """
    for i in range(min(25, len(h_lines) - 1)):
        first_row_crop = img[h_lines[i]:h_lines[i+1], :]
        if first_row_crop.shape[0] < 15: continue
        
        row_resized = cv2.resize(first_row_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(row_resized, config='--psm 7').strip().lower()
        
        # 1. Look for the explicit header
        has_apo = 'apo' in text or 'ap0' in text or 'n°' in text
        has_nom = 'nom' in text
        has_prenom = 'prenom' in text or 'prénom' in text
        
        # Require at least two of the three main column headers to avoid "Nom du module"
        if (has_apo and has_nom) or (has_nom and has_prenom) or (has_apo and has_prenom):
            return h_lines[i]
            
        # 2. Look for a student row (fallback)
        # Students have N_Apo which is a 7 to 10 digit number.
        if re.search(r'\d{7,10}', text):
            # We found a student! The header row MUST be the row right above it.
            return h_lines[max(0, i - 1)]
            
    # Absolute fallback to first line
    return h_lines[0]

def natural_sorted(files):
    def key(f):
        m = re.search(r'(\d+)', f)
        return int(m.group(1)) if m else 0
    return sorted(files, key=key)

def process_document(doc_id):
    prep_dir = f"data/preprocessed/{doc_id}"
    out_dir = f"data/cropped/{doc_id}"
    
    if not os.path.exists(prep_dir):
        print(f"ERROR: Directory not found: {prep_dir}")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    
    img_files = natural_sorted([f for f in glob.glob(f"{prep_dir}/*.jpg") if not f.endswith('_binary.jpg')])
    if not img_files:
        print(f"ERROR: No images found in {prep_dir}")
        return

    for file_idx, img_path in enumerate(img_files):
        filename = os.path.basename(img_path)
        base = os.path.splitext(filename)[0]
        print(f"Processing {filename}...")
        
        img = cv2.imread(img_path)
        h_lines = get_horizontal_lines(img)
        
        if len(h_lines) < 2:
            print(f"  Warning: Could not detect valid horizontal lines in {filename}")
            # Fallback
            cv2.imwrite(f"{out_dir}/{base}_table.jpg", img)
            continue

        if file_idx == 0:
            # First page: Separate header from table
            split_y = find_anchor_line_y(img, h_lines)
            
            if split_y == -1:
                print("  Warning: Could not find 'N° Apo' anchor via OCR. Falling back to the first horizontal line.")
                split_y = h_lines[0]
            
            header = img[:split_y, :]
            table = img[split_y:, :]
            
            cv2.imwrite(f"{out_dir}/{base}_header.jpg", header)
            cv2.imwrite(f"{out_dir}/{base}_table.jpg", table)
            print(f"  Split at Y={split_y}")
        else:
            # Continuation pages are entirely tables
            cv2.imwrite(f"{out_dir}/{base}_table.jpg", img)
            print(f"  Saved as table.")

    print(f"Done. Files saved to {out_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split documents into header and table.")
    parser.add_argument('doc_id', help="Document ID to process (e.g., doc_4)")
    args = parser.parse_args()
    process_document(args.doc_id)