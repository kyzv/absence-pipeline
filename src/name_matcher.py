# src/name_matcher.py
"""
Map OCR‑extracted student names to official group roster using fuzzy matching.
"""

import csv
from typing import Dict, List, Tuple, Optional
from rapidfuzz import process, fuzz

def load_group_roster(csv_path: str) -> List[Dict[str, str]]:
    """
    Read a group CSV (with columns N_Apo, Nom_Prenom) and return a list of dicts.
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        roster = []
        for row in reader:
            napo = row['N_Apo'].strip()
            name = row['Nom_Prenom'].strip()
            if napo and name:
                roster.append({'n_apo': napo, 'nom_prenom': name})
    return roster

def resolve_absences(ocr_names: List[str],
                     group_csv_path: str,
                     min_score: int = 75) -> List[Dict[str, str]]:
    """
    For each OCR‑raw name, find the best matching official name using fuzzy matching.
    Returns a list of {n_apo, nom_prenom, ocr_raw, score}.
    """
    roster = load_group_roster(group_csv_path)
    if not roster:
        return []

    official_names = [r['nom_prenom'] for r in roster]
    results = []
    for raw in ocr_names:
        # fuzzy match against all official names
        match = process.extractOne(raw, official_names, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= min_score:
            best_name = match[0]
            idx = official_names.index(best_name)
            results.append({
                'n_apo': roster[idx]['n_apo'],
                'nom_prenom': roster[idx]['nom_prenom'],
                'ocr_raw': raw,
                'score': match[1]
            })
        else:
            results.append({
                'n_apo': '',
                'nom_prenom': raw,
                'ocr_raw': raw,
                'score': 0
            })
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python src/name_matcher.py <group_csv> <ocr_name1> [ocr_name2 ...]")
        sys.exit(1)
    csv_path = sys.argv[1]
    ocr_names = sys.argv[2:]
    resolved = resolve_absences(ocr_names, csv_path)
    for r in resolved:
        print(f"{r['n_apo']:8s} {r['nom_prenom']:30s} (OCR: {r['ocr_raw']}, score: {r['score']})")