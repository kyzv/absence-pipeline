# src/ocr_header.py
"""
Stage 3 – OCR on the header region.

Uses PaddleOCR 3.x to extract metadata from the binary header.
The OCRResult object is accessed like a dictionary (keys: 'rec_texts', 'rec_scores', 'rec_polys').
"""

import os, re, json
from typing import Dict, List, Optional
from paddleocr import PaddleOCR


def extract_header(image_path: str,
                   output_dir: str = "data/metadata",
                   known_teachers: Optional[List[str]] = None,
                   known_modules: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Extract handwritten/printed metadata from a header image.
    """
    ocr = PaddleOCR(lang='fr', use_textline_orientation=True)

    raw = ocr.predict(image_path)
    if not raw or not raw[0]:
        print("Warning: No text found.")
        return _empty_result()

    page = raw[0]                     # dict-like OCRResult
    # Access the recognized texts, scores, and polygons via dictionary keys
    texts = page['rec_texts']         # list of strings
    scores = page['rec_scores']       # list of floats
    boxes = page['rec_polys']         # list of np.array of shape (4,2)

    # Collect lines with vertical position (top‑left y)
    lines = []
    for box, text, conf in zip(boxes, texts, scores):
        y_top = box[0][1]             # first point's y-coordinate
        lines.append({'text': text.strip(), 'y': y_top, 'confidence': conf})

    # Sort top → bottom
    lines.sort(key=lambda d: d['y'])

    full_text = ' '.join([l['text'] for l in lines])

    enseignant = _extract_field(lines, full_text, r'enseignant\s*[:\-]?\s*(.*)')
    module     = _extract_field(lines, full_text, r'module\s*[:\-]?\s*(.*)')
    element    = _extract_field(lines, full_text, r'[ée]l[ée]ment\s*[:\-]?\s*(.*)')
    date       = _extract_field(lines, full_text, r'date\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})')
    # Heure: we look for "Heure" followed by optional "Début" or "Fin" and capture any digit/h pattern
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

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(output_dir, f"{base}_metadata.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def _extract_field(lines, full_text, pattern, default=''):
    m = re.search(pattern, full_text, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        # Remove any leftover label
        val = re.sub(r'(enseignant|module|élément|element|date|heure)\s*[:\-]?\s*', '', val, flags=re.IGNORECASE).strip()
        if val:
            return val
    # Fallback: next line after keyword
    for i, line in enumerate(lines):
        if re.search(r'enseignant|module|élément|element|date|heure', line['text'], re.IGNORECASE):
            for j in range(i+1, min(i+3, len(lines))):
                cand = lines[j]['text'].strip()
                if cand and not re.search(r'enseignant|module|élément|element|date|heure', cand, re.IGNORECASE):
                    return cand
    return default


def _extract_heure(lines, full_text):
    """
    Extract session time from lines. Handles misread digits like 'Ch30' -> '08h30'.
    """
    # Try to find a pattern like "08h30" or "10h30"
    m = re.search(r'(\d{1,2}[h:]\d{2})', full_text, re.IGNORECASE)
    if m:
        return m.group(1)
    # If no standard pattern, look for something like "Ch30" near "Heure Début"
    for i, line in enumerate(lines):
        if re.search(r'heure\s*d[ée]but', line['text'], re.IGNORECASE):
            for j in range(i+1, min(i+3, len(lines))):
                cand = lines[j]['text'].strip()
                # replace 'Ch' with '08' if it looks like a time
                cand_clean = re.sub(r'\bCh\b', '08', cand)
                if re.search(r'\d{1,2}[h:]\d{2}', cand_clean):
                    return cand_clean
    return ''


def _detect_session_type(lines, full_text):
    # Check for explicit mentions of Cours, TD, TP
    for t in ['Cours', 'TD', 'TP']:
        if re.search(r'\b' + t + r'\b', full_text, re.IGNORECASE):
            return t
    # Also check abbreviations like 'Crs' for Cours
    if re.search(r'\bCrs\b', full_text):
        return 'Cours'
    return ''


def _empty_result():
    return {'enseignant': '', 'module': '', 'element': '', 'date': '', 'heure': '', 'type': ''}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/ocr_header.py <header_image>")
        sys.exit(1)
    meta = extract_header(sys.argv[1])
    print(json.dumps(meta, indent=2, ensure_ascii=False))