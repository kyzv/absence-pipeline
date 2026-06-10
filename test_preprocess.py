import cv2
import numpy as np
import os

def process_image(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image")
        return

    # 1. Correct Orientation
    h, w = img.shape[:2]
    if h > w:
        # Rotate 90 degrees counter-clockwise to make it landscape
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print("Rotated 90 degrees counter-clockwise")

    # Save intermediate orientation
    cv2.imwrite(output_path.replace('.jpg', '_step1_rotated.jpg'), img)

    # 2. Enhance contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
    
    cv2.imwrite(output_path.replace('.jpg', '_step2_enhanced.jpg'), enhanced)

    # 3. Find lines and deskew
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find horizontal lines — use large kernel so only true table lines are detected
    # The table lines span most of the page width; title text is much shorter
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (300, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Find vertical lines — use large kernel for the same reason
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 200))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Find skew angle from horizontal lines
    lines = cv2.HoughLinesP(h_lines, 1, np.pi/180, 200, minLineLength=200, maxLineGap=20)
    angle = 0
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -15 < ang < 15:
                angles.append(ang)
        if angles:
            angle = np.median(angles)
            print(f"Deskew angle: {angle:.2f} degrees")

    # Deskew
    (h, w) = enhanced.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(enhanced, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    
    # Recalculate lines on deskewed image to find bounding box
    gray_deskewed = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
    _, binary_deskewed = cv2.threshold(gray_deskewed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    h_lines_deskewed = cv2.morphologyEx(binary_deskewed, cv2.MORPH_OPEN, h_kernel)
    v_lines_deskewed = cv2.morphologyEx(binary_deskewed, cv2.MORPH_OPEN, v_kernel)
    
    # Dilate slightly to close any small gaps in the detected lines
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    h_lines_deskewed = cv2.dilate(h_lines_deskewed, dilate_k)
    v_lines_deskewed = cv2.dilate(v_lines_deskewed, dilate_k)
    
    table_mask = cv2.add(h_lines_deskewed, v_lines_deskewed)
    
    # Find tight bounding box using pixel projections
    # Sum pixels along each axis to find where lines actually exist
    h_proj = np.sum(table_mask, axis=1)  # Sum each row -> find top/bottom bounds
    v_proj = np.sum(table_mask, axis=0)  # Sum each col -> find left/right bounds

    row_indices = np.where(h_proj > 0)[0]
    col_indices = np.where(v_proj > 0)[0]

    if len(row_indices) > 0 and len(col_indices) > 0:
        y_start = max(0, row_indices[0] - 5)
        y_end   = min(deskewed.shape[0], row_indices[-1] + 5)
        x_start = max(0, col_indices[0] - 5)
        x_end   = min(deskewed.shape[1], col_indices[-1] + 5)
        print(f"Tight crop: x={x_start}:{x_end}, y={y_start}:{y_end}")
        cropped = deskewed[y_start:y_end, x_start:x_end]
    else:
        cropped = deskewed

    cv2.imwrite(output_path, cropped)
    print(f"Saved {output_path}")

os.makedirs("data/debug", exist_ok=True)
process_image("data/raw/doc_1/as1.jpg", "data/debug/test_preprocess.jpg")
