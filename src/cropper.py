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
    Finds the y-coordinate of the bottom line of the 'N° Apo' row.
    This marks the exact split point between Header and Table.
    """
    # Look through the first 15 rows
    for i in range(min(15, len(h_lines) - 1)):
        # Crop the first 800 pixels of the row (which should contain the Apo/Nom text)
        row_crop = img[h_lines[i]:h_lines[i+1], :800]
        row_resized = cv2.resize(row_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        text = pytesseract.image_to_string(row_resized, config='--psm 7').strip().lower()
        if 'apo' in text or 'ap0' in text or 'n°' in text or 'n*' in text:
            # Found the anchor row! The table starts below this row.
            return h_lines[i+1]
            
    return -1

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
                print("  Warning: Could not find 'N° Apo' anchor via OCR. Assuming row 7.")
                # Fallback to row 7 (index 8)
                split_y = h_lines[min(8, len(h_lines)-1)]
            
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