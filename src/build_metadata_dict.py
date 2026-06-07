# src/build_metadata_dict.py
import sys
import os

# Add the project's src directory to the module search path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json, yaml
from pathlib import Path
from expand_dict import expand_name, expand_module, expand_filiere, expand_year

CONFIG_DIR = "data/config"
OUTPUT_PATH = "data/metadata_dict.yaml"

def build():
    # Professors
    with open(Path(CONFIG_DIR) / "professors.json", "r", encoding="utf-8") as f:
        professors = json.load(f)
    teacher_patterns = []
    for p in professors:
        teacher_patterns.extend(expand_name(p))

    # Modules
    with open(Path(CONFIG_DIR) / "modules.json", "r", encoding="utf-8") as f:
        modules = json.load(f)
    mod_patterns = []
    for m in modules:
        mod_patterns.extend(expand_module(m))

    # Filières
    with open(Path(CONFIG_DIR) / "filieres.json", "r", encoding="utf-8") as f:
        filieres = json.load(f)
    fil_patterns = []
    for f in filieres:
        fil_patterns.extend(expand_filiere(f))

    # Years
    with open(Path(CONFIG_DIR) / "years.json", "r", encoding="utf-8") as f:
        years = json.load(f)
    year_patterns = []
    for y in years:
        year_patterns.extend(expand_year(y))

    # Types (hardcoded)
    type_patterns = [
        {"pattern": "(?i)c(ou)?rs?",       "value": "Cours"},
        {"pattern": "(?i)td",              "value": "TD"},
        {"pattern": "(?i)tp",              "value": "TP"},
        {"pattern": "(?i)cnt|contr[ôo]le", "value": "Contrôle"},
        {"pattern": "(?i)ex(am|m)?",       "value": "Examen"}
    ]

    final = {
        "enseignant": teacher_patterns,
        "module":     mod_patterns,
        "filiere":    fil_patterns,
        "annee":      year_patterns,
        "type":       type_patterns
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        yaml.dump(final, f, allow_unicode=True, sort_keys=False)
    print(f"✅ Dictionary built → {OUTPUT_PATH}")

if __name__ == "__main__":
    build()