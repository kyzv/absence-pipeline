"""
ocr_header.py — Extract metadata from the absence sheet header.

Strategy:
  - Tesseract PSM-6  : printed title row (filiere, annee)
  - TrOCR large      : ALL handwritten cells (enseignant, module, date, heure, type)
  - rapidfuzz        : snap every OCR result to its YAML dictionary entry
  - Per-field regex  : date normalisation, time extraction, type matching

Usage:
    python src/ocr_header.py doc_4
    python src/ocr_header.py doc_4 --debug
"""

import os, cv2, json, yaml, re, argparse, logging, warnings
import numpy as np
from rapidfuzz import process, fuzz
import pytesseract
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

warnings.filterwarnings('ignore')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
logging.getLogger('transformers').setLevel(logging.ERROR)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'metadata_dict.yaml')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# ── Lazy TrOCR ────────────────────────────────────────────────────────────────
_processor = None
_model     = None
_device    = None


def get_trocr():
    global _processor, _model, _device
    if _processor is None:
        print("  Loading TrOCR large handwritten model...")
        _device    = "cuda" if torch.cuda.is_available() else "cpu"
        _processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
        _model     = VisionEncoderDecoderModel.from_pretrained(
                         'microsoft/trocr-large-handwritten').to(_device)
        _model.eval()
        print(f"  TrOCR loaded on {_device}.")
    return _processor, _model, _device


# ── Grid detection ────────────────────────────────────────────────────────────

def get_grid_lines(img):
    """Return (h_lines, v_lines) as centre-pixel lists."""
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def proj_lines(morphed, axis, threshold):
        proj = np.sum(morphed, axis=axis)
        lines, in_l, s = [], False, 0
        for i, v in enumerate(proj):
            if v > threshold:
                if not in_l:
                    s = i; in_l = True
            else:
                if in_l:
                    lines.append(int((s + i - 1) / 2)); in_l = False
        if in_l:
            lines.append(int((s + len(proj) - 1) / 2))
        return lines

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (400, 1))
    hm = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    h_lines = proj_lines(hm, axis=1, threshold=255 * 400)

    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    vm = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
    v_lines = proj_lines(vm, axis=0, threshold=255 * 80)

    return h_lines, v_lines


# ── Image helpers ─────────────────────────────────────────────────────────────

def preprocess_crop(crop_bgr, upscale_to_w=600):
    """Trim border, upscale, sharpen. Returns PIL RGB image for TrOCR."""
    h, w = crop_bgr.shape[:2]
    if w < 1 or h < 1:
        return None
    # Trim 2px to avoid grid line bleed
    y1, y2 = min(2, h // 6), max(h - 2, 5 * h // 6)
    x1, x2 = min(2, w // 6), max(w - 2, 5 * w // 6)
    crop_bgr = crop_bgr[y1:y2, x1:x2]
    h, w = crop_bgr.shape[:2]
    if h < 4 or w < 4:
        return None
    # Upscale
    if w < upscale_to_w:
        scale    = upscale_to_w / w
        crop_bgr = cv2.resize(crop_bgr, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
    # Mild sharpening
    kernel   = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    crop_bgr = cv2.filter2D(crop_bgr, -1, kernel)
    return Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))


def is_blank(crop_bgr, threshold=0.97):
    """True if >threshold fraction of pixels are white (cell is empty)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return True
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    return (np.sum(bw == 255) / bw.size) > threshold


# ── TrOCR inference ───────────────────────────────────────────────────────────

def trocr_read(crop_bgr, upscale_to_w=600):
    """Run TrOCR on a BGR crop. Returns (text, confidence)."""
    if is_blank(crop_bgr):
        return "", 1.0
    pil_img = preprocess_crop(crop_bgr, upscale_to_w=upscale_to_w)
    if pil_img is None:
        return "", 0.0
    proc, model, device = get_trocr()
    try:
        pixel_values = proc(images=pil_img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=32,
            )
        text = proc.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
        if outputs.scores:
            probs = [torch.softmax(s, dim=-1).max().item() for s in outputs.scores]
            conf  = float(np.mean(probs))
        else:
            conf = 0.5
        return text, round(conf, 3)
    except Exception as e:
        print(f"    TrOCR error: {e}")
        return "", 0.0


# ── Tesseract (printed text only) ────────────────────────────────────────────

def tesseract_read(crop_bgr, psm=6):
    """Read printed text with Tesseract."""
    if crop_bgr is None or crop_bgr.shape[0] < 5:
        return ""
    up = cv2.resize(crop_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return pytesseract.image_to_string(up, config=f'--psm {psm} -l fra').strip()


# ── Fuzzy matching ────────────────────────────────────────────────────────────

def fuzzy_match(raw, choices, min_score=50):
    """Snap raw OCR text to closest YAML entry. Returns (value, conf 0-1)."""
    valid = [c for c in (choices or []) if c and str(c).strip()]
    if not raw or not raw.strip():
        return "", 1.0
    best = process.extractOne(raw, valid, scorer=fuzz.WRatio)
    if best:
        val, score, _ = best
        conf = round(score / 100.0, 2)
        return (val, conf) if score >= min_score else (raw, conf)
    return raw, 0.0


# ── Field normalisation ───────────────────────────────────────────────────────

def normalize_date(raw):
    """
    Extract dd/mm/yyyy from noisy TrOCR output.
    Falls back to digit-group extraction when separators are garbled.
    """
    if not raw or not raw.strip():
        return "", 1.0
    if not re.search(r'\d', raw):
        return "", 0.0

    # Helper to validate and format date parts
    def _try_parts(d, mo, y):
        if len(y) > 4:
            y = y[-2:]
        if len(y) == 2:
            prefix = config.get('date', {}).get('year_century_prefix', 20)
            y = str(prefix) + y
        try:
            if 1 <= int(d) <= 31 and 1 <= int(mo) <= 12 and 2000 <= int(y) <= 2099:
                return f"{d.zfill(2)}/{mo.zfill(2)}/{y}"
        except ValueError:
            pass
        return None

    # Try direct regex first (TrOCR usually reads slashes correctly)
    patterns = config.get('date', {}).get('patterns', [r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}'])
    for p in patterns:
        m = re.search(p, raw)
        if m:
            date_str = m.group(0)
            parts = re.split(r'[/\-.|\\]', date_str)
            if len(parts) == 3:
                result = _try_parts(parts[0], parts[1], parts[2])
                if result:
                    return result, 0.95

    # Fallback: extract all digit groups, assume dd/mm/yy order
    digits = re.findall(r'\d+', raw)

    if len(digits) >= 3:
        result = _try_parts(digits[0], digits[1], digits[2])
        if result:
            return result, 0.7
        # Special case: first group too large (e.g. '17103' from '17(103')
        g0 = digits[0]
        if len(g0) > 2:
            result = _try_parts(g0[:2], g0[2:], digits[1])
            if result:
                return result, 0.65

    elif len(digits) == 2:
        g0, g1 = digits[0], digits[1]
        # Year embedded in second group (e.g. '07 4126' = 07/04/2026)
        if len(g1) >= 4:
            result = _try_parts(g0, g1[:2], g1[2:])
            if result:
                return result, 0.55
        # Plain dd/mm with unknown year
        try:
            if 1 <= int(g0) <= 31 and 1 <= int(g1) <= 12:
                return f"{g0.zfill(2)}/{g1.zfill(2)}/?", 0.3
        except ValueError:
            pass

    return "", 0.0


def normalize_time(raw, choices):
    """Extract hXXhMM from raw and snap to YAML list.

    Applies common TrOCR character substitutions before parsing:
    - 'p'/'P' after digit → '4'  (e.g. '1ph40' → '14h40')
    - Leading 'At'/'Ath'  → '14' (e.g. 'Athoo' → '14h00')
    - Trailing letter noise after digit+o → '00'
    """
    if not raw or not raw.strip():
        return "", 1.0

    # OCR character substitutions
    cleaned = raw
    cleaned = re.sub(r'(\d)[pP]', r'\g<1>4', cleaned)           # '1ph40' → '14h40'
    cleaned = re.sub(r'^At?([hH])', r'14\1', cleaned)             # 'Athoo' → '14hoo'
    cleaned = re.sub(r'^A(\d)', r'14\1', cleaned)                 # 'A4h00' → '144h00'
    cleaned = re.sub(r'(\d)[oO][a-zA-Z]\s*$', r'\g<1>00', cleaned) # '14hos' → '14h00' (after h already)
    cleaned = re.sub(r'(\d)thorn', r'\1h00', cleaned, flags=re.IGNORECASE)  # '16thorn'→'16h00'
    cleaned = re.sub(r'thorn', 'h00', cleaned, flags=re.IGNORECASE)          # bare 'thorn'→'h00'

    digit_count = len(re.findall(r'\d', cleaned))
    has_h = bool(re.search(r'[hH:]', cleaned))
    if digit_count < 1 or (not has_h and digit_count < 3):
        return "", 0.0

    m = re.search(r'(\d{1,2})\s*[hH:]\s*(\d{0,2})', cleaned)
    if m:
        h    = m.group(1)
        mins = (m.group(2) or '00').ljust(2, '0')
        cand = f"{h}h{mins}"
        valid = [c for c in (choices or []) if c]
        best  = process.extractOne(cand, valid, scorer=fuzz.WRatio)
        if best:
            val, score, _ = best
            if score >= 70:
                return val, round(score / 100.0, 2)
        return cand, 0.5
    return "", 0.0


def normalize_type(raw, choices):
    """Match session type: crs/tp/td/cnt/exm."""
    if not raw or not raw.strip():
        return "", 1.0
    low = raw.lower().replace(' ', '')
    for t in (choices or []):
        if t and t.lower() in low:
            return t, 0.95
    return fuzzy_match(raw, choices, min_score=55)


# ── Title-row parsing (Tesseract on printed text) ─────────────────────────────

def parse_title_row(img, h_lines):
    """Read the printed 'Filiere ... YYYY-YYYY' line above the grid."""
    y_top = max(0, h_lines[0] - 10)
    y_bot = h_lines[1] + 10 if len(h_lines) > 1 else img.shape[0]
    crop  = img[y_top:y_bot, :]
    text  = tesseract_read(crop, psm=6)

    filiere_val, filiere_conf = "", 0.0
    annee_val,   annee_conf   = "", 0.0

    m = re.search(r'Fili.re\s+(.*?)\s*[-\u2013]\s*(\d{4}[-\u2013]\d{4})', text, re.IGNORECASE)
    if m:
        filiere_raw  = m.group(1).strip()
        annee_raw    = m.group(2).replace('\u2013', '-')
        filiere_val, filiere_conf = fuzzy_match(filiere_raw, config.get('filieres', []))
        annee_val,   annee_conf   = fuzzy_match(annee_raw,   config.get('annee',    []))
    else:
        m2 = re.search(r'(\d{4}[-\u2013]\d{4})', text)
        if m2:
            annee_raw  = m2.group(1).replace('\u2013', '-')
            annee_val, annee_conf = fuzzy_match(annee_raw, config.get('annee', []))
        before = re.sub(r'\d{4}.*', '', text)
        raw_f  = re.sub(r'.*(?:Fili.re|pr.sence\s*:)', '', before, flags=re.IGNORECASE).strip()
        if raw_f:
            filiere_val, filiere_conf = fuzzy_match(raw_f, config.get('filieres', []))

    return (filiere_val, filiere_conf), (annee_val, annee_conf)


# ── Main processing ───────────────────────────────────────────────────────────

def process_document(doc_id, debug=False):
    img_path = os.path.join('data', 'cropped', doc_id, 'as1_header.jpg')
    out_dir  = os.path.join('data', 'output')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(img_path):
        print(f"ERROR: {img_path} not found.")
        return

    print(f"[ocr_header] Processing {doc_id}...")
    img = cv2.imread(img_path)
    h_lines, v_lines = get_grid_lines(img)

    if len(h_lines) < 5 or len(v_lines) < 5:
        print(f"ERROR: Grid detection failed (h={len(h_lines)}, v={len(v_lines)}).")
        return

    # Find where seance columns begin (narrow cols < 350px)
    seance_start_col = 2
    for c in range(1, len(v_lines) - 1):
        if (v_lines[c + 1] - v_lines[c]) < 350:
            seance_start_col = c
            break

    num_sessions = len(v_lines) - 1 - seance_start_col
    print(f"  Grid: {len(h_lines)} h-lines, {len(v_lines)} v-lines, "
          f"{num_sessions} seances, label_cols=0..{seance_start_col-1}")

    # ── Output skeleton ───────────────────────────────────────────────────────
    out = {
        "doc_id":     doc_id,
        "filiere":    {"value": "", "confidence": 0.0},
        "annee":      {"value": "", "confidence": 0.0},
        "enseignant": {"value": "", "confidence": 0.0},
        "module":     {"value": "", "confidence": 0.0},
        "seances":    {},
        "absences":   []
    }
    for s in range(1, num_sessions + 1):
        out["seances"][f"seance{s}"] = {
            "date":        {"value": "", "confidence": 1.0},
            "heure_debut": {"value": "", "confidence": 1.0},
            "heure_fin":   {"value": "", "confidence": 1.0},
            "type":        {"value": "", "confidence": 1.0},
        }

    # ── 1. Printed title row → filiere + annee ────────────────────────────────
    (fv, fc), (av, ac) = parse_title_row(img, h_lines)
    out["filiere"] = {"value": fv, "confidence": fc}
    out["annee"]   = {"value": av, "confidence": ac}

    # ── 2. Identify rows by printed label (Tesseract, left 300px) ────────────
    row_map = {}
    for i in range(1, len(h_lines) - 1):
        y1, y2 = h_lines[i], h_lines[i + 1]
        lbl    = ""
        for vx in [v_lines[0], v_lines[1] if len(v_lines) > 2 else v_lines[0]]:
            crop_lbl = img[y1:y2, vx: vx + 300]
            lbl += ' ' + tesseract_read(crop_lbl, psm=7).lower()

        if 'enseignant' in lbl:                            row_map['enseignant']  = i
        elif 'module'   in lbl:                            row_map['module']      = i
        elif 'date'     in lbl and 'date' not in row_map:  row_map['date']        = i
        elif 'but'      in lbl:                            row_map['heure_debut'] = i
        elif 'fin'      in lbl:                            row_map['heure_fin']   = i
        elif 'type'     in lbl or 'crs' in lbl:            row_map['type']        = i

    print(f"  Row mapping: {row_map}")

    # ── 3. Enseignant (TrOCR, full width of value cell) ───────────────────────
    if 'enseignant' in row_map:
        ri     = row_map['enseignant']
        y1, y2 = h_lines[ri], h_lines[ri + 1]
        crop   = img[y1:y2, v_lines[1]: img.shape[1]]
        raw, ocr_conf   = trocr_read(crop, upscale_to_w=800)
        val, match_conf = fuzzy_match(raw, config.get('enseignants', []))
        out['enseignant'] = {
            "value":      val if val else raw,
            "confidence": round((ocr_conf + match_conf) / 2, 2) if raw else 1.0,
        }
        if debug:
            print(f"    enseignant raw='{raw}' -> '{out['enseignant']['value']}'")

    # ── 4. Module (TrOCR, up to seance col start) ────────────────────────────
    if 'module' in row_map:
        ri     = row_map['module']
        y1, y2 = h_lines[ri], h_lines[ri + 1]
        crop   = img[y1:y2, v_lines[1]: v_lines[seance_start_col]]
        raw, ocr_conf   = trocr_read(crop, upscale_to_w=800)
        val, match_conf = fuzzy_match(raw, config.get('modules', []))
        out['module'] = {
            "value":      val if val else raw,
            "confidence": round((ocr_conf + match_conf) / 2, 2) if raw else 1.0,
        }
        if debug:
            print(f"    module raw='{raw}' -> '{out['module']['value']}'")

    # ── 5. Per-seance fields ──────────────────────────────────────────────────
    for s_idx in range(num_sessions):
        s_num = s_idx + 1
        s_key = f"seance{s_num}"
        cx1   = v_lines[seance_start_col + s_idx]
        cx2   = v_lines[seance_start_col + s_idx + 1]

        def cell(row_key, _cx1=cx1, _cx2=cx2):
            if row_key not in row_map:
                return None
            ri = row_map[row_key]
            return img[h_lines[ri]:h_lines[ri + 1], _cx1:_cx2]

        # DATE
        crop = cell('date')
        if crop is not None:
            raw, ocr_conf = trocr_read(crop, upscale_to_w=400)
            norm, d_conf  = normalize_date(raw)
            out['seances'][s_key]['date'] = {
                "value":      norm,
                "confidence": round(ocr_conf * d_conf, 2) if raw else 1.0,
            }
            if debug:
                print(f"    s{s_num} date raw='{raw}' -> '{norm}'")

        # HEURE DEBUT
        crop = cell('heure_debut')
        if crop is not None:
            raw, ocr_conf = trocr_read(crop, upscale_to_w=400)
            norm, t_conf  = normalize_time(raw, config.get('heure_debut', []))
            out['seances'][s_key]['heure_debut'] = {
                "value":      norm,
                "confidence": round(ocr_conf * t_conf, 2) if raw else 1.0,
            }
            if debug:
                print(f"    s{s_num} heure_debut raw='{raw}' -> '{norm}'")

        # HEURE FIN
        crop = cell('heure_fin')
        if crop is not None:
            raw, ocr_conf = trocr_read(crop, upscale_to_w=400)
            norm, t_conf  = normalize_time(raw, config.get('heure_fin', []))
            out['seances'][s_key]['heure_fin'] = {
                "value":      norm,
                "confidence": round(ocr_conf * t_conf, 2) if raw else 1.0,
            }
            if debug:
                print(f"    s{s_num} heure_fin raw='{raw}' -> '{norm}'")

        # TYPE
        crop = cell('type')
        if crop is not None:
            raw, ocr_conf = trocr_read(crop, upscale_to_w=400)
            norm, tp_conf = normalize_type(raw, config.get('types', []))
            out['seances'][s_key]['type'] = {
                "value":      norm,
                "confidence": round(ocr_conf * tp_conf, 2) if raw else 1.0,
            }
            if debug:
                print(f"    s{s_num} type raw='{raw}' -> '{norm}'")

    # ── 6. Write JSON ─────────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, f"{doc_id}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=4, ensure_ascii=False)
    print(f"  [OK] Written to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('doc_id', help='e.g. doc_4')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    process_document(args.doc_id, debug=args.debug)