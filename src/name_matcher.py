"""src/name_matcher.py"""

import os
import json
import re
import pandas as pd
import unicodedata
from typing import Dict, List, Tuple, Optional
from rapidfuzz import fuzz, distance

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    # remove accents
    text = "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    # replace non-alphanumeric with spaces
    text = re.sub(r'[^a-z0-9]', ' ', text)
    return " ".join(text.split())

def map_filiere_to_csv(filiere_name: str, config_dir: str = "config/groups", mapping_path: str = "config/filiere_mapping.json") -> str:
    """
    Fuzzy matches the filiere name to the filenames in config/groups/ using the mapping json.
    """
    if not filiere_name:
        return ""
    
    # Load mapping dictionary if it exists
    mapping = {}
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
        except Exception as e:
            print(f"Error loading mapping file {mapping_path}: {e}")
            
    # Clean the input filiere name
    cleaned_input = clean_text(filiere_name)
    
    # Check exact or substring clean match in the mapping keys
    for key, csv_file in mapping.items():
        cleaned_key = clean_text(key)
        if cleaned_key == cleaned_input or cleaned_key in cleaned_input or cleaned_input in cleaned_key:
            return csv_file
            
    # Use rapidfuzz to find the best match amongst the keys
    if mapping:
        from rapidfuzz import process
        keys = list(mapping.keys())
        best_match = process.extractOne(filiere_name, keys, scorer=fuzz.WRatio)
        if best_match:
            matched_key, score, _ = best_match
            if score > 50:
                return mapping[matched_key]
                
    # Fallback to listing csv files in groups and matching directly against their names
    if os.path.exists(config_dir):
        csv_files = [f for f in os.listdir(config_dir) if f.endswith('.csv')]
        if csv_files:
            clean_files = {clean_text(os.path.splitext(f)[0]): f for f in csv_files}
            from rapidfuzz import process
            best_match = process.extractOne(cleaned_input, list(clean_files.keys()), scorer=fuzz.WRatio)
            if best_match:
                matched_clean, score, _ = best_match
                if score > 50:
                    return clean_files[matched_clean]
            return csv_files[0]
            
    return ""

def match_student(ocr_text: str, student_df: pd.DataFrame) -> Tuple[Optional[Dict], float]:
    """
    Matches OCR text row (containing digits and names) to a student in the student_df.
    Returns (matched_student_dict, confidence_score 0-1).
    """
    ocr_clean = clean_text(ocr_text)
    ocr_digits = "".join(c for c in ocr_text if c.isdigit())
    
    best_student = None
    best_score = -1.0
    
    # Identify the columns dynamically
    col_map = {col.lower().strip(): col for col in student_df.columns}
    
    id_col = next((col_map[k] for k in ['n_apo', 'apo', 'id', 'n°'] if k in col_map), student_df.columns[0])
    nom_col = next((col_map[k] for k in ['nom'] if k in col_map), None)
    prenom_col = next((col_map[k] for k in ['prenom', 'prénom'] if k in col_map), None)
    
    # If nom/prenom columns are not explicitly separated, try to find any column with "nom" or "name"
    if nom_col is None:
        nom_col = next((col for col in student_df.columns if 'nom' in col.lower() or 'name' in col.lower() or 'etudiant' in col.lower()), student_df.columns[1] if len(student_df.columns) > 1 else student_df.columns[0])
        
    for _, row in student_df.iterrows():
        # Get values
        n_apo_val = str(row[id_col]).strip()
        nom_val = str(row[nom_col]).strip() if nom_col else ""
        prenom_val = str(row[prenom_col]).strip() if prenom_col else ""
        
        # Format candidate representations
        full_student_str = clean_text(f"{n_apo_val} {nom_val} {prenom_val}")
        name_only_str = clean_text(f"{nom_val} {prenom_val}")
        
        # Calculate matching scores
        # 1. Fuzzy token sorting ratio on clean text
        token_sort_ratio = fuzz.token_sort_ratio(ocr_clean, full_student_str)
        ratio_full = fuzz.ratio(ocr_clean, full_student_str)
        
        # 2. Match Apogée ID (using Levenshtein distance on digits)
        id_score = 0.0
        if ocr_digits and n_apo_val:
            lev_dist = distance.Levenshtein.distance(ocr_digits, n_apo_val)
            max_len = max(len(ocr_digits), len(n_apo_val))
            id_score = (1.0 - lev_dist / max_len) * 100.0 if max_len > 0 else 0.0
            
        # 3. Fuzzy partial ratio on names only
        name_ratio = fuzz.partial_ratio(ocr_clean, name_only_str)
        
        # Combined confidence score heuristic
        if id_score >= 80:
            combined_score = 0.7 * id_score + 0.3 * token_sort_ratio
        else:
            combined_score = 0.2 * id_score + 0.8 * max(ratio_full, token_sort_ratio, name_ratio)
            
        if combined_score > best_score:
            best_score = combined_score
            best_student = {
                "n_apo": n_apo_val,
                "nom": nom_val,
                "prenom": prenom_val,
                "fullname": f"{nom_val} {prenom_val}".strip()
            }
            
    confidence = round(best_score / 100.0, 3)
    return best_student, confidence

def match_names(absent_indices: List[int], student_db_path: str) -> List[str]:
    """Deprecated: keeps backward compatibility."""
    df = pd.read_csv(student_db_path)
    col = _find_name_column(df)
    names = df[col].tolist()
    return [names[i] for i in absent_indices if 0 <= i < len(names)]

def _find_name_column(df: pd.DataFrame) -> str:
    keywords = ['name', 'nom', 'prenom', 'prénom', 'student', 'étudiant']
    for col in df.columns:
        col_lower = col.lower().replace('&', '').replace('et ', '').strip()
        for kw in keywords:
            if kw in col_lower:
                return col
    return df.columns[0]
