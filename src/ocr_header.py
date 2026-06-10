import os
import cv2
import glob
import json
import yaml
import re
import argparse
import numpy as np
from rapidfuzz import process, fuzz
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import warnings

warnings.filterwarnings('ignore')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YAML Config
with open("config/metadata_dict.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Global variables for model lazy-loading
processor = None
model = None

def init_trocr():
    global processor, model
    if processor is None:
        print("Loading TrOCR large model... (this may take a moment)")
        # We use CPU by default unless cuda is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten').to(device)
        print(f"TrOCR loaded on {device}.")

def get_grid_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Horizontal lines
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

    # Vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    v_morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    v_proj = np.sum(v_morphed, axis=0)
    
    v_lines = []
    in_line = False; start_x = 0
    for x, val in enumerate(v_proj):
        if val > 255 * 80:
            if not in_line: start_x = x; in_line = True
        else:
            if in_line: v_lines.append(int((start_x + x - 1) / 2)); in_line = False
    if in_line: v_lines.append(int((start_x + len(v_proj) - 1) / 2))

    return h_lines, v_lines

def extract_trocr(img_crop):
    init_trocr()
    # Convert BGR to RGB
    rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pixel_values = processor(images=pil_img, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_length=64)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

def extract_tesseract(img_crop, psm=7):
    # Upscale for better tesseract reading
    upscaled = cv2.resize(img_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(upscaled, config=f'--psm {psm} -l fra').strip()
    return text

def fuzzy_match(text, choices):
    if not text.strip():
        return "", 1.0
        
    res = process.extractOne(text, choices, scorer=fuzz.WRatio)
    if res:
        match_str, score, _ = res
        confidence = score / 100.0
        if confidence >= 0.8:
            return match_str, round(confidence, 2)
        elif confidence >= 0.5:
            return match_str, round(confidence, 2)
        else:
            return text, round(confidence, 2)
    return text, 0.0

def fuzzy_match_time(text, choices):
    if not text.strip():
        return "", 1.0
    
    # Extract digit-h-digit pattern
    match = re.search(r"(\d{1,2})[hH:](\d{0,2})", text.lower())
    if match:
        h = match.group(1)
        m = match.group(2)
        if not m: m = "00"
        if len(m) == 1: m = m + "0"
        canonical = f"{h}h{m}"
        
        res = process.extractOne(canonical, choices, scorer=fuzz.WRatio)
        if res:
            match_str, score, _ = res
            return match_str, round(score / 100.0, 2)
        return canonical, 0.8
    return text, 0.0

def normalize_date(text):
    if not text.strip():
        return "", 1.0
    
    patterns = config['date']['patterns']
    for p in patterns:
        m = re.search(p, text)
        if m:
            raw_date = m.group(0)
            raw_date = raw_date.replace('-', '/').replace('|', '/')
            parts = raw_date.split('/')
            if len(parts) == 3:
                d, m_, y = parts
                if len(y) == 2:
                    y = str(config['date']['year_century_prefix']) + y
                return f"{d.zfill(2)}/{m_.zfill(2)}/{y}", 0.95
                
    return text, 0.0

def process_document(doc_id):
    img_path = f"data/cropped/{doc_id}/as1_header.jpg"
    out_dir = f"data/output"
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(img_path):
        print(f"ERROR: File not found: {img_path}")
        return
        
    print(f"Processing header for {doc_id}...")
    img = cv2.imread(img_path)
    h_lines, v_lines = get_grid_lines(img)
    
    if len(h_lines) < 5 or len(v_lines) < 5:
        print("ERROR: Grid detection failed.")
        return

    # Total columns. Usually ~11-13
    num_cols = len(v_lines) - 1
    
    # We find the start of the séance columns by checking their widths.
    # Label columns are wide (e.g., > 400px). Séance columns are narrow (~200px).
    seance_start_col = 2
    for c in range(1, len(v_lines)-1):
        if (v_lines[c+1] - v_lines[c]) < 400:
            seance_start_col = c
            break
            
    num_sessions = num_cols - seance_start_col

    # Output structure
    out_data = {
        "doc_id": doc_id,
        "filiere": {"value": "", "confidence": 0.0},
        "annee": {"value": "", "confidence": 0.0},
        "enseignant": {"value": "", "confidence": 0.0},
        "module": {"value": "", "confidence": 0.0},
        "seances": {},
        "absences": []
    }
    
    # Initialize seance objects
    for s in range(1, num_sessions + 1):
        out_data["seances"][f"seance{s}"] = {
            "date": {"value": "", "confidence": 1.0},
            "heure_debut": {"value": "", "confidence": 1.0},
            "heure_fin": {"value": "", "confidence": 1.0},
            "type": {"value": "", "confidence": 1.0}
        }

    # 1. Parse Title Row (Filière + Année)
    title_crop = img[h_lines[0]-10:h_lines[1]+10, :]
    title_text = extract_tesseract(title_crop, psm=6)
    
    # Regex: Filière (.*) - (\d{4}-\d{4})
    match = re.search(r"Filire (.*?)\s*-\s*(\d{4}-\d{4})", title_text.replace('è', ''))
    if match:
        filiere_raw = match.group(1).strip()
        annee_raw = match.group(2).strip()
        
        f_val, f_conf = fuzzy_match(filiere_raw, config['filieres'])
        a_val, a_conf = fuzzy_match(annee_raw, config['annee'])
        
        out_data["filiere"] = {"value": f_val, "confidence": f_conf}
        out_data["annee"] = {"value": a_val, "confidence": a_conf}
    else:
        # Fallback regex
        match2 = re.search(r"Fili.re (.*?)\s*-\s*(\d{4}-\d{4})", title_text)
        if match2:
            filiere_raw = match2.group(1).strip()
            annee_raw = match2.group(2).strip()
            f_val, f_conf = fuzzy_match(filiere_raw, config['filieres'])
            a_val, a_conf = fuzzy_match(annee_raw, config['annee'])
            out_data["filiere"] = {"value": f_val, "confidence": f_conf}
            out_data["annee"] = {"value": a_val, "confidence": a_conf}

    # 2. Iterate over rows to find specific fields
    row_mapping = {}
    for i in range(1, len(h_lines)-1):
        # 1. Check the very left edge (first column)
        crop1 = img[h_lines[i]:h_lines[i+1], v_lines[0]:v_lines[0]+250]
        text1 = extract_tesseract(crop1).lower()
        
        # 2. Check the inset edge (second column)
        text2 = ""
        if len(v_lines) > 1:
            crop2 = img[h_lines[i]:h_lines[i+1], v_lines[1]:v_lines[1]+250]
            text2 = extract_tesseract(crop2).lower()
            
        label_text = text1 + " " + text2

        if "enseignant" in label_text:
            row_mapping["enseignant"] = i
        elif "module" in label_text:
            row_mapping["module"] = i
        elif "date" in label_text:
            row_mapping["date"] = i
        elif "but" in label_text:
            row_mapping["heure_debut"] = i
        elif "fin" in label_text:
            row_mapping["heure_fin"] = i
        elif "type" in label_text or "crs" in label_text:
            row_mapping["type"] = i

    print("Detected Row Mapping:", row_mapping)

    # 3. Extract Hand-written values using TrOCR
    
    # Enseignant
    if "enseignant" in row_mapping:
        row_i = row_mapping["enseignant"]
        # Enseignant value is everything to the right of the label cell
        val_crop = img[h_lines[row_i]:h_lines[row_i+1], v_lines[1]:v_lines[-1]]
        val_text = extract_trocr(val_crop)
        val, conf = fuzzy_match(val_text, config['enseignants'])
        out_data["enseignant"] = {"value": val, "confidence": conf}

    # Module
    if "module" in row_mapping:
        row_i = row_mapping["module"]
        # Module value spans until the "Element" column usually, we can just crop the first large block
        # Usually v_lines[1] to v_lines[seance_start_col]
        val_crop = img[h_lines[row_i]:h_lines[row_i+1], v_lines[1]:v_lines[seance_start_col]]
        val_text = extract_trocr(val_crop)
        val, conf = fuzzy_match(val_text, config['modules'])
        out_data["module"] = {"value": val, "confidence": conf}

    # Séances
    for s_idx in range(num_sessions):
        s_num = s_idx + 1
        col_start = v_lines[seance_start_col + s_idx]
        col_end = v_lines[seance_start_col + s_idx + 1]
        
        # Date
        if "date" in row_mapping:
            row_i = row_mapping["date"]
            val_crop = img[h_lines[row_i]:h_lines[row_i+1], col_start:col_end]
            # Ignore tiny crops
            if (h_lines[row_i+1] - h_lines[row_i]) > 10 and (col_end - col_start) > 10:
                val_text = extract_trocr(val_crop)
                val, conf = normalize_date(val_text)
                # Ensure we handle empty correctly
                if val == "": conf = 1.0
                out_data["seances"][f"seance{s_num}"]["date"] = {"value": val, "confidence": conf}
                
        # Heure Debut
        if "heure_debut" in row_mapping:
            row_i = row_mapping["heure_debut"]
            val_crop = img[h_lines[row_i]:h_lines[row_i+1], col_start:col_end]
            if (h_lines[row_i+1] - h_lines[row_i]) > 10 and (col_end - col_start) > 10:
                val_text = extract_trocr(val_crop)
                val, conf = fuzzy_match_time(val_text, config['heure_debut'])
                if val == "": conf = 1.0
                out_data["seances"][f"seance{s_num}"]["heure_debut"] = {"value": val, "confidence": conf}

        # Heure Fin
        if "heure_fin" in row_mapping:
            row_i = row_mapping["heure_fin"]
            val_crop = img[h_lines[row_i]:h_lines[row_i+1], col_start:col_end]
            if (h_lines[row_i+1] - h_lines[row_i]) > 10 and (col_end - col_start) > 10:
                val_text = extract_trocr(val_crop)
                val, conf = fuzzy_match_time(val_text, config['heure_fin'])
                if val == "": conf = 1.0
                out_data["seances"][f"seance{s_num}"]["heure_fin"] = {"value": val, "confidence": conf}

        # Type
        if "type" in row_mapping:
            row_i = row_mapping["type"]
            val_crop = img[h_lines[row_i]:h_lines[row_i+1], col_start:col_end]
            if (h_lines[row_i+1] - h_lines[row_i]) > 10 and (col_end - col_start) > 10:
                val_text = extract_trocr(val_crop)
                val_text_lower = val_text.lower().replace(' ', '')
                
                # exact matching for types
                matched = False
                for t in config['types']:
                    if t and t.lower() in val_text_lower:
                        out_data["seances"][f"seance{s_num}"]["type"] = {"value": t, "confidence": 0.95}
                        matched = True
                        break
                
                if not matched:
                    if val_text == "":
                        out_data["seances"][f"seance{s_num}"]["type"] = {"value": "", "confidence": 1.0}
                    else:
                        # try fuzzy
                        val, conf = fuzzy_match(val_text, config['types'])
                        out_data["seances"][f"seance{s_num}"]["type"] = {"value": val, "confidence": conf}

    # Write JSON
    out_json = f"{out_dir}/{doc_id}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=4, ensure_ascii=False)
        
    print(f"Successfully wrote {out_json}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract header metadata using OCR.")
    parser.add_argument('doc_id', help="Document ID to process (e.g., doc_4)")
    args = parser.parse_args()
    process_document(args.doc_id)