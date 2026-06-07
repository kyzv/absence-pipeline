# src/ocr_header.py
"""
Stage 3 of the absence pipeline — OCR on the header region.

Responsibilities:
    1. Load the cropped header image (from cropper.py).
    2. Run PaddleOCR to detect and recognise all text.
    3. Parse the recognised text to extract:
        - Enseignant (teacher)
        - Module
        - Elément
        - Date
        - Heure
        - Type de séance (Cours / TD / TP)
    4. Return a structured dictionary and optionally save it as JSON.

This module handles both printed text (labels like "Enseignant :") and
handwritten text (the teacher's entries). No checkbox detection is done
here – that belongs to checkbox_detection.py.

Input  : path to header image (e.g., data/cropped/as1_header.jpg)
Output : dict with keys 'enseignant','module','element','date','heure','type'
         + optional JSON file saved in data/metadata/
"""

import os
import re
import json
from typing import Dict, List, Optional
from paddleocr import PaddleOCR


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def extract_header(image_path: str,
                   output_dir: str = "data/metadata",
                   known_teachers: Optional[List[str]] = None,
                   known_modules: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Extract handwritten/printed metadata from a header image.

    Args:
        image_path      : path to the header image (BGR, from cropper.py).
        output_dir      : folder where the extracted JSON will be saved.
        known_teachers  : optional list of valid teacher names (for fuzzy
                          correction – not yet implemented).
        known_modules   : optional list of valid module names.

    Returns:
        {
            'enseignant': '...',
            'module':     '...',
            'element':    '...',
            'date':       '...',
            'heure':      '...',
            'type':       'Cours' / 'TD' / 'TP' / ''
        }
    """
    # ── 1. Initialise PaddleOCR ──────────────────────────────────────────
    # PaddleOCR(use_angle_cls=True, lang='fr') :
    #   use_angle_cls=True → a small model corrects upside-down or rotated text
    #   lang='fr'          → French language model (improves accent handling)
    #   show_log=False     → suppress progress bars for cleaner output
    ocr = PaddleOCR(use_textline_orientation=True, lang='fr')

    # ── 2. Run OCR ──────────────────────────────────────────────────────
    # ocr.ocr(image_path) returns a list of pages. Since we pass a single
    # image, we get a list with one element: [ [ [box, (text, confidence)], ... ] ].
    result = ocr.predict(image_path)
    if not result or not result[0]:
        print("Warning: No text found in header image.")
        return _empty_result()

    # Flatten: we only have one page → result[0]
    detections = result[0]   # list of [box, (text, confidence)]

    # ── 3. Collect all recognised texts ─────────────────────────────────
    # We'll keep the full text and its vertical position (y‑coordinate)
    # because the header fields appear in a predictable top‑to‑bottom order.
    lines = []
    for det in detections:
        box, (text, conf) = det
        # box is a list of 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        # We use the top‑left y to sort lines.
        y_top = box[0][1]
        lines.append({
            'text': text.strip(),
            'y': y_top,
            'confidence': conf
        })

    # Sort top → bottom
    lines.sort(key=lambda d: d['y'])

    # ── 4. Extract fields with simple rules ─────────────────────────────
    # The header layout we assume (based on the images):
    #   - A line containing "Enseignant" followed by the teacher's name
    #   - A line containing "Module" followed by the module name
    #   - A line containing "Elément" (or "Element") followed by the element name
    #   - Date and Heure may appear as "Date : dd/mm/yyyy" and "Heure : HH:MM"
    #   - Type de séance (Cours/TD/TP) may be written next to checkboxes or as text
    #
    # Because OCR mixes labels and handwritten text, we'll search for
    # keywords and extract whatever follows them.

    full_text = ' '.join([l['text'] for l in lines])

    enseignant = _extract_field(lines, full_text, r'enseignant\s*[:\-]?\s*(.*)', default='')
    module     = _extract_field(lines, full_text, r'module\s*[:\-]?\s*(.*)', default='')
    element    = _extract_field(lines, full_text, r'[ée]l[ée]ment\s*[:\-]?\s*(.*)', default='')
    date       = _extract_field(lines, full_text, r'date\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})', default='')
    heure      = _extract_field(lines, full_text, r'heure\s*[:\-]?\s*(\d{1,2}[h:]\d{2})', default='')
    session_type = _detect_session_type(lines, full_text)

    metadata = {
        'enseignant': enseignant,
        'module': module,
        'element': element,
        'date': date,
        'heure': heure,
        'type': session_type
    }

    # ── 5. Save to JSON ─────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(output_dir, f"{base}_metadata.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_field(lines: List[Dict], full_text: str, pattern: str, default: str = '') -> str:
    """
    Try to extract a field using a regular expression.
    First search the full concatenated text. If no match, try individual lines.
    """
    match = re.search(pattern, full_text, re.IGNORECASE)
    if match:
        value = match.group(1).strip()
        # Remove possible leftover label fragments
        value = re.sub(r'(enseignant|module|élément|element|date|heure)\s*[:\-]?\s*', '', value, flags=re.IGNORECASE).strip()
        if value:
            return value

    # Fallback: examine lines near known keywords
    for i, line in enumerate(lines):
        if re.search(r'enseignant|module|élément|element|date|heure', line['text'], re.IGNORECASE):
            # The next line(s) might contain the handwritten value
            for j in range(i+1, min(i+3, len(lines))):
                candidate = lines[j]['text'].strip()
                if candidate and not re.search(r'enseignant|module|élément|element|date|heure', candidate, re.IGNORECASE):
                    return candidate
    return default


def _detect_session_type(lines: List[Dict], full_text: str) -> str:
    """
    Determine whether Cours, TD, or TP was selected.
    Looks for checkmarks near the words 'Cours', 'TD', 'TP' or explicit text.
    Returns 'Cours', 'TD', 'TP' or ''.
    """
    # Simple text-based detection: if the word appears with high confidence
    # and not as a label, we assume it's the selected one.
    types = ['Cours', 'TD', 'TP']
    for t in types:
        if re.search(r'\b' + t + r'\b', full_text, re.IGNORECASE):
            # Check if there is a nearby checkmark – but for now we just return
            # the first occurrence that isn't obviously a label.
            # A more advanced version would use checkbox detection.
            return t
    return ''


def _empty_result() -> Dict[str, str]:
    return {
        'enseignant': '',
        'module': '',
        'element': '',
        'date': '',
        'heure': '',
        'type': ''
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND‑LINE TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/ocr_header.py <header_image>")
        sys.exit(1)

    header_path = sys.argv[1]
    meta = extract_header(header_path)
    print(json.dumps(meta, indent=2, ensure_ascii=False))