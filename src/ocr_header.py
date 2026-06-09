# src/ocr_header.py
import re
from typing import Dict, List, Tuple
from thefuzz import process, fuzz

def extract_document_metadata(ocr_blocks: List[Dict]) -> Dict:
    """
    Extracts specific metadata fields from OCR text blocks found in the header area.
    Expected fields: filiere, annee, enseignant, module, date, heure_debut, heure_fin, type
    """
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
                        # Extract what's after the keyword
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
                # Ensure it doesn't look like another header key
                text = block['text'].lower()
                if dist < min_dist and not any(kw in text for kw in ["filiere", "filière", "enseignant", "module", "date", "heure", "séance", "seance"]):
                    min_dist = dist
                    best_match = block['text']
                    
        return best_match

    # 1. Module
    metadata["module"] = find_closest_value(["module"])

    # 2. Enseignant
    metadata["enseignant"] = find_closest_value(["enseignant", "prof", "pr."])

    # 3. Filiere & Annee (often together like "Filiere Genie Info - 2024-2025")
    filiere_val = find_closest_value(["filière", "filiere"])
    if filiere_val:
        # Try to extract year from it
        year_match = re.search(r'(\d{4}\s*-\s*\d{4})', filiere_val)
        if year_match:
            metadata["annee"] = year_match.group(1).replace(" ", "")
            filiere_val = filiere_val.replace(year_match.group(0), "").strip(" -_")
        metadata["filiere"] = filiere_val
    else:
        # Try finding year separately
        metadata["annee"] = find_closest_value(["année", "annee"])

    # 4. Date
    metadata["date"] = find_closest_value(["date", "le :", "le:"])

    # 5. Heure (Debut & Fin)
    # Sometimes it's written as "Heure: 08h30 à 10h30" or "Heure début: ... Heure fin: ..."
    heure_debut = find_closest_value(["heure début", "heure debut", "de :", "de:"])
    heure_fin = find_closest_value(["heure fin", "à :", "a :", "à:", "a:"])
    
    if not heure_debut or not heure_fin:
        # Check if there is a general "Horaire" or "Heure"
        horaire = find_closest_value(["heure", "horaire"])
        if horaire:
            # Try to split by 'à' or '-'
            parts = re.split(r'(?i)\s*(?:à|a|-|/|au)\s*', horaire)
            if len(parts) >= 2:
                heure_debut = parts[0]
                heure_fin = parts[1]
            else:
                heure_debut = horaire
                
    metadata["heure_debut"] = heure_debut
    metadata["heure_fin"] = heure_fin

    # 6. Type (crs, tp, td, cnt, exm)
    # Find any text block that looks like these, or find the closest to "Type"
    type_val = find_closest_value(["type", "séance", "seance"])
    valid_types = ["crs", "tp", "td", "cnt", "exm", "cours", "controle", "examen"]
    
    if type_val:
        # Fuzzy match the extracted type against our valid types
        best_type, score = process.extractOne(type_val.lower(), valid_types, scorer=fuzz.ratio)
        if score > 60:
            if best_type == "cours": best_type = "crs"
            elif best_type == "controle": best_type = "cnt"
            elif best_type == "examen": best_type = "exm"
            metadata["type"] = best_type
    
    # If spatial matching failed for type, just scan all blocks for the checkboxes near these words
    if not metadata["type"]:
        for block in ocr_blocks:
            text = block['text'].lower().strip()
            # If a block is exactly one of the types
            best_type, score = process.extractOne(text, valid_types, scorer=fuzz.ratio)
            if score > 85: # High confidence it's exactly the label
                if best_type == "cours": best_type = "crs"
                elif best_type == "controle": best_type = "cnt"
                elif best_type == "examen": best_type = "exm"
                metadata["type"] = best_type
                break

    return metadata