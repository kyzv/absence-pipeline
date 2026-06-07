import os, re, json
from typing import Dict, Optional
from paddleocr import PaddleOCR

# ----------------------------------------------------------------------
# PUBLIC – process one document
# ----------------------------------------------------------------------
def extract_document_metadata(doc_id: str,
                              cropped_root: str = "data/cropped",
                              metadata_root: str = "data/metadata") -> Dict[str, str]:
    crop_dir = os.path.join(cropped_root, doc_id)
    header_path = os.path.join(crop_dir, "as1_header.jpg")
    if not os.path.exists(header_path):
        headers = [f for f in os.listdir(crop_dir) if "_header.jpg" in f]
        if headers:
            header_path = os.path.join(crop_dir, headers[0])
        else:
            raise FileNotFoundError(f"No header image found in {crop_dir}")

    ocr = PaddleOCR(lang='fr', use_textline_orientation=True)
    raw = ocr.predict(header_path)
    if not raw or not raw[0]:
        return _empty_result()

    page = raw[0]
    texts = page['rec_texts']
    scores = page['rec_scores']
    boxes = page['rec_polys']

    lines = []
    for box, text, conf in zip(boxes, texts, scores):
        y_top = box[0][1]
        lines.append({'text': text.strip(), 'y': y_top, 'confidence': conf})
    lines.sort(key=lambda d: d['y'])

    full_text = ' '.join([l['text'] for l in lines])

    enseignant = _extract_field(lines, full_text, r'enseignant\s*[:\-]?\s*(.*)')
    module     = _extract_field(lines, full_text, r'module\s*[:\-]?\s*(.*)')
    element    = _extract_field(lines, full_text, r'[ée]l[ée]ment\s*[:\-]?\s*(.*)')
    date       = _extract_field(lines, full_text, r'date\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})')
    heure      = _extract_heure(lines, full_text)
    session_type = _detect_session_type(lines, full_text)

    metadata = {
        'enseignant': enseignant,
        'module': module,
        'element': element,
        'date': date,
        'heure': heure,
        'type': session_type
    }

    out_dir = os.path.join(metadata_root, doc_id)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "metadata.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def _extract_field(lines, full_text, pattern, default=''):
    m = re.search(pattern, full_text, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        val = re.sub(r'(enseignant|module|élément|element|date|heure)\s*[:\-]?\s*', '', val, flags=re.IGNORECASE).strip()
        if val:
            return val
    for i, line in enumerate(lines):
        if re.search(r'enseignant|module|élément|element|date|heure', line['text'], re.IGNORECASE):
            for j in range(i+1, min(i+3, len(lines))):
                cand = lines[j]['text'].strip()
                if cand and not re.search(r'enseignant|module|élément|element|date|heure', cand, re.IGNORECASE):
                    return cand
    return default

def _extract_heure(lines, full_text):
    m = re.search(r'(\d{1,2}[h:]\d{2})', full_text, re.IGNORECASE)
    if m:
        return m.group(1)
    for i, line in enumerate(lines):
        if re.search(r'heure\s*d[ée]but', line['text'], re.IGNORECASE):
            for j in range(i+1, min(i+3, len(lines))):
                cand = lines[j]['text'].strip()
                cand_clean = re.sub(r'\bCh\b', '08', cand)
                if re.search(r'\d{1,2}[h:]\d{2}', cand_clean):
                    return cand_clean
    return ''

def _detect_session_type(lines, full_text):
    for t in ['Cours', 'TD', 'TP']:
        if re.search(r'\b' + t + r'\b', full_text, re.IGNORECASE):
            return t
    if re.search(r'\bCrs\b', full_text):
        return 'Cours'
    return ''

def _empty_result():
    return {'enseignant': '', 'module': '', 'element': '', 'date': '', 'heure': '', 'type': ''}


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/ocr_header.py <doc_id>")
        sys.exit(1)
    doc_id = sys.argv[1]
    meta = extract_document_metadata(doc_id)
    print(json.dumps(meta, indent=2, ensure_ascii=False))