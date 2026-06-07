# src/ocr_header.py
import os, json, re
from typing import Dict, List
import yaml
import numpy as np
from paddleocr import PaddleOCR


def extract_document_metadata(doc_id: str,
                              cropped_root: str = "data/cropped",
                              metadata_root: str = "data/metadata",
                              template_path: str = "data/expected_metadata.json",
                              dict_path: str = "data/metadata_dict.yaml") -> Dict:
    # ---- 1. Load template and dictionary ----
    with open(template_path, 'r', encoding='utf-8') as f:
        template = json.load(f)
    with open(dict_path, 'r', encoding='utf-8') as f:
        correction_dict = yaml.safe_load(f)

    # ---- 2. Locate header image ----
    crop_dir = os.path.join(cropped_root, doc_id)
    header_path = os.path.join(crop_dir, "as1_header.jpg")
    if not os.path.exists(header_path):
        headers = [f for f in os.listdir(crop_dir) if "_header.jpg" in f]
        if headers:
            header_path = os.path.join(crop_dir, headers[0])
        else:
            raise FileNotFoundError(f"No header image found in {crop_dir}")

    # ---- 3. Run OCR ----
    ocr = PaddleOCR(lang='fr', use_textline_orientation=True)
    raw = ocr.predict(header_path)
    if not raw or not raw[0]:
        return template   # return empty template

    page = raw[0]
    texts = page['rec_texts']
    boxes = page['rec_polys']

    entries = []
    for box, text in zip(boxes, texts):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        entries.append({
            'text': text.strip(),
            'x': sum(xs) / len(xs),
            'y': sum(ys) / len(ys)
        })
    entries.sort(key=lambda e: (e['y'], e['x']))

    # ---- 4. Extract top-level fields ----
    filiere_raw, annee_raw = _extract_filiere_annee(entries)
    enseignant_raw = _extract_simple_field(entries, 'enseignant')
    module_raw     = _extract_simple_field(entries, 'module')

    filiere    = _fuzzy_correct(filiere_raw, 'filiere', correction_dict)
    annee      = _fuzzy_correct(annee_raw, 'annee', correction_dict)
    enseignant = _fuzzy_correct(enseignant_raw, 'enseignant', correction_dict)
    module     = _fuzzy_correct(module_raw, 'module', correction_dict)

    # ---- 5. Build séance table ----
    col_map = {}
    for e in entries:
        m = re.match(r'^séance(\d{1,2})$', e['text'].lower())
        if m:
            col_map[int(m.group(1))] = e['x']

    row_bands = _build_row_bands(entries)
    seances = template.get('seances', {
        str(i): {"date": "", "heure_debut": "", "heure_fin": "", "type": ""}
        for i in range(1, 11)
    })

    if col_map and row_bands:
        value_entries = [e for e in entries if not _is_grid_label(e['text'])]
        for e in value_entries:
            col = _closest_column(e['x'], col_map)
            row = _row_for_y(e['y'], row_bands)
            if col and row:
                key = _row_to_key(row)
                col_str = str(col)
                if col_str in seances and seances[col_str].get(key, '') == '':
                    raw_val = e['text']
                    if key == 'type':
                        raw_val = _fuzzy_correct(raw_val, 'type', correction_dict)
                    seances[col_str][key] = raw_val

    metadata = {
        "filiere": filiere,
        "annee": annee,
        "enseignant": enseignant,
        "module": module,
        "seances": seances
    }

    # ---- 6. Save ----
    out_dir = os.path.join(metadata_root, doc_id)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "metadata.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


# ── Helpers ────────────────────────────────────────────────────────────

def _extract_filiere_annee(entries):
    for e in entries:
        m = re.search(r'Filière\s+(.+?)\s*-\s*(\d{4}-\d{4})', e['text'], re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return '', ''

def _extract_simple_field(entries, label):
    for i, e in enumerate(entries):
        if e['text'].lower().startswith(label.lower()):
            for j in range(i+1, min(i+3, len(entries))):
                txt = entries[j]['text']
                if txt and not _is_ignored(txt) and not _is_grid_label(txt):
                    return txt
    return ''

def _is_ignored(text):
    low = text.lower()
    phrases = [
        "université sultan moulay slimane",
        "l'ecole supérieure de technologie",
        "l'école supérieure de technologie",
        "fquih ben salah",
        "liste de présence",
        "n.b:",
        "a : absence",
        "p : présence",
        "n apo",
        "nom & prenom",
    ]
    return any(p in low for p in phrases)

def _is_grid_label(text):
    low = text.lower()
    if re.match(r'^séance\d{1,2}$', low): return True
    if low in ('date', 'heure début', 'heure fin'): return True
    if low.startswith('type'): return True
    return _is_ignored(text)

def _build_row_bands(entries):
    positions = {}
    for e in entries:
        low = e['text'].lower()
        if low == 'date':           positions['date'] = e['y']
        elif 'heure début' in low:  positions['heure_debut'] = e['y']
        elif 'heure fin' in low:    positions['heure_fin'] = e['y']
        elif low.startswith('type'): positions['type'] = e['y']
    sorted_labels = sorted(positions.items(), key=lambda kv: kv[1])
    bands = {}
    for i, (name, y) in enumerate(sorted_labels):
        next_y = sorted_labels[i+1][1] if i+1 < len(sorted_labels) else float('inf')
        bands[name] = (y, next_y)
    return bands

def _closest_column(x, col_map):
    best, best_dist = None, float('inf')
    for num, cx in col_map.items():
        dist = abs(x - cx)
        if dist < best_dist:
            best, best_dist = num, dist
    return best

def _row_for_y(y, bands):
    for name, (low, high) in bands.items():
        if low <= y <= high:
            return name
    return None

def _row_to_key(row):
    if row == 'date': return 'date'
    if row in ('heure_debut', 'heure début'): return 'heure_debut'
    if row in ('heure_fin', 'heure fin'): return 'heure_fin'
    if row == 'type': return 'type'
    return row

def _fuzzy_correct(raw_text, field, dictionary):
    patterns = dictionary.get(field, [])
    for entry in patterns:
        if re.search(entry['pattern'], raw_text):
            return entry['value']
    return raw_text


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/ocr_header.py <doc_id>")
        sys.exit(1)
    doc_id = sys.argv[1]
    meta = extract_document_metadata(doc_id)
    print(json.dumps(meta, indent=2, ensure_ascii=False))