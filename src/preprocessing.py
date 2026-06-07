# src/preprocessing.py
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple


def preprocess_document(doc_id: str,
                        raw_root: str = "data/raw",
                        preprocessed_root: str = "data/preprocessed") -> List[Dict]:
    """
    Clean and binarize every page of a document.

    1. Reads all images from data/raw/<doc_id>/.
    2. Sorts them naturally (as1, as2, …).
    3. For each image: deskew, enhance contrast, binarize.
    4. Saves the cleaned colour image and a binary version to
       data/preprocessed/<doc_id>/.

    Returns a list of dicts (one per page) with keys:
        'original', 'cleaned', 'binary', 'output_path', 'binary_path'
    """
    raw_dir = os.path.join(raw_root, doc_id)
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Raw document folder not found: {raw_dir}")

    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = _natural_sorted([
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(exts)
    ])

    if not image_files:
        raise FileNotFoundError(f"No image files found in {raw_dir}")

    results = []
    for img_file in image_files:
        img_path = os.path.join(raw_dir, img_file)
        res = _preprocess_one(img_path, doc_id, preprocessed_root)
        results.append(res)

    return results


# ----------------------------------------------------------------------
# SINGLE-PAGE LOGIC
# ----------------------------------------------------------------------
def _preprocess_one(image_path: str, doc_id: str, output_root: str) -> Dict:
    image = _load(image_path)
    gray = _to_grayscale(image)
    deskewed = _deskew(image, gray)
    enhanced = _enhance(deskewed)
    binary = _binarize(enhanced)

    out_dir = os.path.join(output_root, doc_id)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    clean_path = os.path.join(out_dir, f"{base}.jpeg")
    bin_path   = os.path.join(out_dir, f"{base}_binary.jpg")
    cv2.imwrite(clean_path, enhanced)
    cv2.imwrite(bin_path, binary)

    return {
        'original': image,
        'cleaned': enhanced,
        'binary': binary,
        'output_path': clean_path,
        'binary_path': bin_path,
    }


# ----------------------------------------------------------------------
# HELPERS (page extraction removed)
# ----------------------------------------------------------------------
def _load(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img

def _to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def _deskew(image: np.ndarray, gray: np.ndarray, max_skew_angle: float = 10.0) -> np.ndarray:
    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    merged = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, h_kernel)
    lines = cv2.HoughLinesP(merged, 1, np.pi / 180, threshold=150, minLineLength=80, maxLineGap=15)
    if lines is None:
        return image
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < max_skew_angle:
            angles.append(angle)
    if not angles:
        return image
    skew_angle = float(np.median(angles))
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _enhance(image: np.ndarray, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

def _binarize(image: np.ndarray, block_size: int = 31, C: int = 15) -> np.ndarray:
    gray = _to_grayscale(image) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary_inv = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C)
    return cv2.bitwise_not(binary_inv)

def _natural_sorted(files: List[str]) -> List[str]:
    import re
    def key(f):
        m = re.search(r'(\d+)', f)
        return int(m.group(1)) if m else 0
    return sorted(files, key=key)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/preprocessing.py <doc_id>")
        sys.exit(1)
    doc_id = sys.argv[1]
    results = preprocess_document(doc_id)
    for r in results:
        print(f"Preprocessed: {os.path.basename(r['output_path'])}")