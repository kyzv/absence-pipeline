# src/ocr_header.py
import re
import yaml
from typing import Dict, List, Tuple
from thefuzz import process, fuzz
import os

def load_metadata_dict(dict_path: str = "data/metadata_dict.yaml") -> Dict:
    if not os.path.exists(dict_path):
        return {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def fuzzy_correct(value: str, choices: List[str], threshold: int = 60) -> str:
    if not value or not choices:
        return value
    match, score = process.extractOne(value, choices, scorer=fuzz.token_set_ratio)
    if score >= threshold:
        return match
    return value

def extract_document_metadata(ocr_blocks: List[Dict]) -> Dict:
    """
    Extracts specific metadata fields from OCR text blocks found in the header area.
    """
    metadata_dict = load_metadata_dict()
    
    metadata = {
        "filiere": "",
        "annee": "",
        "enseignant": "",
        "module": "",
        "date": "",
        "heure_debut": "",
        "heure_fin": "",
        "type": ""
    }
    
    if not ocr_blocks:
        return metadata

    # Find value strictly to the right or below an anchor text
    def find_closest_value(anchor_keywords: List[str], max_dist_x=400, max_dist_y=50) -> str:
        anchor_block = None
        # Find the anchor
        for block in ocr_blocks:
            text = block['text'].lower()
            if any(kw in text for kw in anchor_keywords):
                # If the anchor text already contains the value (e.g. "Module : Math")
                for kw in anchor_keywords:
                    if kw in text and len(text) > len(kw) + 3:
                        match = re.split(rf'(?i){kw}\s*:?\s*', text)
                        if len(match) > 1 and match[1].strip():
                            return match[1].strip()
                anchor_block = block
                break
                
        if not anchor_block:
            return ""
            
        ax, ay = anchor_block['center']
        best_match = ""
        min_dist = float('inf')
        
        for block in ocr_blocks:
            if block == anchor_block:
                continue
            bx, by = block['center']
            dx = bx - ax
            dy = by - ay
            
            # Look right (dx > 0 and small dy) or bottom (dy > 0 and small dx)
            if (0 < dx < max_dist_x and abs(dy) < 30) or (0 < dy < max_dist_y and abs(dx) < 100):
                dist = dx**2 + dy**2
                text = block['text'].lower()
                if dist < min_dist and not any(kw in text for kw in ["filiere", "filière", "enseignant", "module", "date", "heure", "séance", "seance"]):
                    min_dist = dist
                    best_match = block['text']
                    
        return best_match

    # Extract raw values
    raw_module = find_closest_value(["module"])
    raw_enseignant = find_closest_value(["enseignant", "prof", "pr."])
    
    filiere_val = find_closest_value(["filière", "filiere"])
    raw_annee = ""
    raw_filiere = ""
    
    if filiere_val:
        year_match = re.search(r'(\d{4}\s*-\s*\d{4})', filiere_val)
        if year_match:
            raw_annee = year_match.group(1).replace(" ", "")
            raw_filiere = filiere_val.replace(year_match.group(0), "").strip(" -_")
        else:
            raw_filiere = filiere_val
    else:
        raw_annee = find_closest_value(["année", "annee"])

    raw_date = find_closest_value(["date", "le :", "le:"])

    heure_debut = find_closest_value(["heure début", "heure debut", "de :", "de:"])
    heure_fin = find_closest_value(["heure fin", "à :", "a :", "à:", "a:"])
    
    if not heure_debut or not heure_fin:
        horaire = find_closest_value(["heure", "horaire"])
        if horaire:
            parts = re.split(r'(?i)\s*(?:à|a|-|/|au)\s*', horaire)
            if len(parts) >= 2:
                heure_debut = parts[0]
                heure_fin = parts[1]
            else:
                heure_debut = horaire
                
    raw_type = find_closest_value(["type", "séance", "seance"])
    
    # ---------------------------------------------------------
    # DICTIONARY CORRECTION
    # ---------------------------------------------------------
    metadata["module"] = fuzzy_correct(raw_module, metadata_dict.get('modules', [])) if raw_module else ""
    metadata["enseignant"] = fuzzy_correct(raw_enseignant, metadata_dict.get('enseignants', [])) if raw_enseignant else ""
    metadata["filiere"] = fuzzy_correct(raw_filiere, metadata_dict.get('filieres', [])) if raw_filiere else ""
    metadata["annee"] = raw_annee
    metadata["date"] = raw_date
    metadata["heure_debut"] = heure_debut
    metadata["heure_fin"] = heure_fin
    
    # Clean Type
    valid_types = metadata_dict.get('types', ["crs", "tp", "td", "cnt", "exm"])
    if raw_type:
        best_type, score = process.extractOne(raw_type.lower(), valid_types + ["cours", "controle", "examen"], scorer=fuzz.ratio)
        if score > 60:
            if best_type == "cours": best_type = "crs"
            elif best_type == "controle": best_type = "cnt"
            elif best_type == "examen": best_type = "exm"
            metadata["type"] = best_type
    
    if not metadata["type"]:
        for block in ocr_blocks:
            text = block['text'].lower().strip()
            best_type, score = process.extractOne(text, valid_types + ["cours", "controle", "examen"], scorer=fuzz.ratio)
            if score > 85:
                if best_type == "cours": best_type = "crs"
                elif best_type == "controle": best_type = "cnt"
                elif best_type == "examen": best_type = "exm"
                metadata["type"] = best_type
                break

    return metadata