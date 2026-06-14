import os
import cv2
import glob
import numpy as np
import argparse
import re

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

def natural_sorted(files):
    def key(f):
        m = re.search(r'(\d+)', f)
        return int(m.group(1)) if m else 0
    return sorted(files, key=key)

def process_document(doc_id):
    cropped_dir = f"data/cropped/{doc_id}"
    out_dir = f"data/rows/{doc_id}"
    
    if not os.path.exists(cropped_dir):
        print(f"ERROR: Directory not found: {cropped_dir}")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    
    # Get all _table.jpg files, naturally sorted to maintain page order
    table_files = natural_sorted(glob.glob(f"{cropped_dir}/*_table.jpg"))
    if not table_files:
        print(f"ERROR: No table images found in {cropped_dir}")
        return

    global_row_counter = 1

    for img_path in table_files:
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Error reading {filename}")
            continue

        h_lines = get_horizontal_lines(img)
        
        if len(h_lines) < 2:
            print(f"  Warning: Could not detect valid horizontal lines in {filename}. Skipping.")
            continue

        is_first_valid_row = True
        for i in range(len(h_lines) - 1):
            y_start = h_lines[i]
            y_end = h_lines[i+1]
            
            row_height = y_end - y_start
            
            # Filter out tiny spurious lines or massive gaps
            # Normal row heights are typically 50-80 pixels
            if row_height < 20 or row_height > 200:
                continue
                
            row_img = img[y_start:y_end, :]
            
            # Check if this first row is a header row (N Apo, Nom, Prenom)
            if is_first_valid_row:
                import pytesseract
                padded_row = cv2.copyMakeBorder(row_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                # Remove INTER_CUBIC because it breaks OCR on some images (e.g. doc_4)
                row_resized = cv2.resize(padded_row, None, fx=2, fy=2)
                text = pytesseract.image_to_string(row_resized, config='--psm 7').strip().lower()
                
                # If text is completely empty, it might be a spurious border line or artifact.
                # Drop it, and KEEP is_first_valid_row = True to check the next row!
                if not text:
                    print(f"  Dropped empty line/artifact from {filename}")
                    continue
                    
                is_first_valid_row = False
                
                # If there's an 8 digit Apo number, it is DEFINITELY a student.
                if not re.search(r'\d{7,10}', text):
                    # It doesn't have a student number. Is it the header?
                    # Check for any of the header words.
                    has_apo = 'apo' in text or 'ap0' in text or 'n°' in text
                    has_nom = 'nom' in text
                    has_prenom = 'prenom' in text or 'prénom' in text
                    has_seance = 'sean' in text or 'séan' in text or 'san' in text
                    
                    if has_apo or has_nom or has_prenom or has_seance:
                        print(f"  Dropped header row from {filename} (Text: {text[:30]})")
                        continue
            
            # Save the row
            out_path = f"{out_dir}/row_{global_row_counter:03d}.jpg"
            cv2.imwrite(out_path, row_img)
            global_row_counter += 1

    print(f"Done. Sliced {global_row_counter - 1} total rows into {out_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Slice table images into individual row images.")
    parser.add_argument('doc_id', help="Document ID to process (e.g., doc_4)")
    args = parser.parse_args()
    process_document(args.doc_id)
