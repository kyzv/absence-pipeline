"""
src/preprocessing.py

This file contains all the "Computer Vision" tools used to clean up the scanned image 
before the program tries to read text or detect checkboxes.
"""

import os
import cv2          # OpenCV: The main library for image processing
import numpy as np  # Numpy: A library for doing fast math on large grids of numbers (like images)
from typing import Dict, List, Tuple


def preprocess_document(doc_id: str,
                        raw_root: str = "data/raw",
                        preprocessed_root: str = "data/preprocessed") -> List[Dict]:
    """
    Clean and binarize every page of a document.

    1. Reads all images from data/raw/<doc_id>/.
    2. Sorts them naturally (as1, as2, ...).
    3. For each image: deskew, enhance contrast, binarize.
    4. Saves the cleaned colour image and a binary version to
       data/preprocessed/<doc_id>/.

    Returns a list of dicts (one per page) with keys:
        'original', 'cleaned', 'binary', 'output_path', 'binary_path'
    """
    
    # Create the full path to the raw images folder
    raw_dir = os.path.join(raw_root, doc_id)
    
    # Check if the folder actually exists on the hard drive
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Raw document folder not found: {raw_dir}")

    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    # List Comprehension: "Find all files 'f' in the raw_dir IF their name ends with one of the allowed extensions"
    # We then wrap it in _natural_sorted so "page2" comes before "page10".
    image_files = _natural_sorted([
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(exts)
    ])

    if not image_files:
        raise FileNotFoundError(f"No image files found in {raw_dir}")

    results = []
    # Loop through each found image file one by one
    for img_file in image_files:
        img_path = os.path.join(raw_dir, img_file)
        
        # Call our helper function to do the actual cleaning work for this single image
        res = _preprocess_one(img_path, doc_id, preprocessed_root)
        
        # Add the result to our list
        results.append(res)

    return results


# ----------------------------------------------------------------------
# SINGLE-PAGE LOGIC
# ----------------------------------------------------------------------
def _preprocess_one(image_path: str, doc_id: str, output_root: str) -> Dict:
    """
    This function takes a single image, applies a sequence of cleaning steps,
    saves the cleaned versions, and returns a dictionary with the results.
    """
    image = load_image(image_path)
    deskewed = deskew(image)
    enhanced = enhance(deskewed)
    binary = binarize(enhanced)

    # Prepare to save the results
    out_dir = os.path.join(output_root, doc_id)
    os.makedirs(out_dir, exist_ok=True) # Create output folder if it doesn't exist
    
    # Get just the file name without the extension (e.g., "scan1.jpg" -> "scan1")
    base = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create the new file names
    clean_path = os.path.join(out_dir, f"{base}.jpeg")
    bin_path   = os.path.join(out_dir, f"{base}_binary.jpg")
    
    # Save (write) the images to the hard drive
    cv2.imwrite(clean_path, enhanced)
    cv2.imwrite(bin_path, binary)

    # Return the images and paths as a dictionary
    return {
        'original': image,
        'cleaned': enhanced,
        'binary': binary,
        'output_path': clean_path,
        'binary_path': bin_path,
    }


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return img

def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

def deskew(image: np.ndarray, max_skew_angle: float = 10.0) -> np.ndarray:
    """Straightens the image based on text line angles."""
    gray = to_grayscale(image)
    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Dilate horizontally to merge text into lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    merged = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, h_kernel)
    
    lines = cv2.HoughLinesP(merged, 1, np.pi / 180, threshold=150, minLineLength=80, maxLineGap=15)
    
    if lines is None:
        return image
        
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < max_skew_angle and abs(angle) > 0.1:
            angles.append(angle)
            
    if not angles:
        return image
        
    skew_angle = float(np.median(angles))
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    
    # Use white background for borders (255, 255, 255)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return deskewed

def crop_margins(image: np.ndarray, margin: int = 30) -> np.ndarray:
    """Removes empty white space around the printed document content."""
    gray = to_grayscale(image)
    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Use morphology to connect all text into big blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    closed = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
        
    x_min, y_min = image.shape[1], image.shape[0]
    x_max, y_max = 0, 0
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h > 20: # Ignore tiny noise
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
            
    if x_max <= x_min or y_max <= y_min:
        return image
        
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(image.shape[1], x_max + margin)
    y_max = min(image.shape[0], y_max + margin)
    
    return image[y_min:y_max, x_min:x_max]

def enhance(image: np.ndarray, clip_limit: float = 2.0, tile_size: tuple = (8, 8)) -> np.ndarray:
    """Improves contrast using CLAHE."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

def binarize(image: np.ndarray, block_size: int = 31, C: int = 15) -> np.ndarray:
    """
    Adaptive thresholding to return a clean black and white image.
    Ink will be BLACK (0) and background will be WHITE (255) because of bitwise_not.
    """
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # THRESH_BINARY_INV makes ink 255 (white) and background 0 (black)
    binary_inv = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C)
    # bitwise_not makes ink 0 (black) and background 255 (white)
    return cv2.bitwise_not(binary_inv)

def _natural_sorted(files: List[str]) -> List[str]:
    """
    A small helper that sorts numbers inside text correctly.
    Standard sorting: image1.jpg, image10.jpg, image2.jpg
    Natural sorting: image1.jpg, image2.jpg, image10.jpg
    """
    import re
    def key(f):
        # Look for digits (\d+) in the filename
        m = re.search(r'(\d+)', f)
        return int(m.group(1)) if m else 0
    return sorted(files, key=key)


# ----------------------------------------------------------------------
# This part only runs if you execute this specific file from the terminal
# e.g., "python src/preprocessing.py my_doc"
if __name__ == "__main__":
    import sys
    # Check if the user provided an argument (the document ID)
    if len(sys.argv) < 2:
        print("Usage: python src/preprocessing.py <doc_id>")
        sys.exit(1) # Stop the program
        
    doc_id = sys.argv[1]
    results = preprocess_document(doc_id)
    
    for r in results:
        # Print a success message for each file processed
        print(f"Preprocessed: {os.path.basename(r['output_path'])}")