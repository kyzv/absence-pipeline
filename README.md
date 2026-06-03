# Automatisation de la Saisie des Absences
### Pipeline de Reconnaissance de Documents et Vision par Ordinateur

---

## Contexte

Dans le cadre de la digitalisation des processus administratifs de l'école, la gestion manuelle des feuilles d'absence représente une tâche chronophage et sujette aux erreurs de saisie. Les feuilles de présence sont pré-imprimées avec la liste nominale des étudiants. L'enseignant remplit manuellement l'en-tête (module, nom, type de séance, date, horaire) et coche les cases des étudiants absents.

Ce projet propose un pipeline intelligent capable de **scanner une feuille d'absence, d'en extraire automatiquement les métadonnées manuscrites, et d'identifier les étudiants absents** afin d'automatiser leur saisie dans le système d'information de l'école.

---

## Aperçu du Pipeline

```
Image brute
    └─► preprocessing.py    → Image binaire nettoyée
         └─► ocr_header.py  → Métadonnées de l'en-tête {module, enseignant, date, horaire,type séance}
              └─► checkbox_detection.py → Indices des lignes absentes
                   └─► name_matcher.py  → Noms des étudiants absents
                        └─► pipeline.py → Résultat structuré 
                             └─► app.py → Interface web Streamlit
```

---

## Structure du Projet

```
absence-pipeline/
├── data/
│   ├── raw/              # Dataset feuilles d'absence scannées originales
│   ├── annotated/        # Annotations LabelMe (zones en-tête / tableau)
│   └── students.csv      # Liste officielle des étudiants (colonne : name)
├── src/
│   ├── preprocessing.py       # Étape 1 : Nettoyage de l'image (OpenCV)
│   ├── ocr_header.py          # Étape 2 : Extraction de l'en-tête (PaddleOCR)
│   ├── checkbox_detection.py  # Étape 3 : Détection des absences (densité pixels)
│   ├── name_matcher.py        # Étape 4 : Correspondance des noms (Levenshtein)
│   └── pipeline.py            # Orchestration des étapes 1 à 4
├── app.py                # Interface web Streamlit
├── requirements.txt      # Dépendances Python
└── README.md             # Documentation du projet
```

---

## Prérequis

- Python **3.12** 
- pip 25+

---

## Installation

**1. Cloner le dépôt**
```bash
git clone https://github.com/kyzv/absence-pipeline.git
cd absence-pipeline
```

**2. Installer les dépendances**
```bash
pip install -r requirements.txt
```

**3. Vérifier l'installation**
```bash
python -c "import cv2; from paddleocr import PaddleOCR; import streamlit; from rapidfuzz import fuzz; import pandas; import numpy; print('Toutes les dépendances sont installées correctement.')"
```

---

## Utilisation

### Lancer l'interface web
```bash
streamlit run app.py
```
L'application s'ouvre automatiquement dans le navigateur à `http://localhost:8501`.

### Utiliser le pipeline en ligne de commande
```python
from src.pipeline import run

result = run(
    image_path="data/raw/sheet_001.jpg",
    student_db_path="data/students.csv"
)
print(result)
```

### Format du fichier `students.csv`
```csv
name
BENNANI Omar
CHERKAOUI Fatima
SAKHI Mehdi
...
```
L'ordre des noms doit correspondre exactement à l'ordre d'impression sur la feuille d'absence.

---

## Description des Modules

### `src/preprocessing.py`
Nettoie l'image brute en quatre étapes séquentielles :
- **Chargement** — lecture du fichier image en mémoire via OpenCV
- **Conversion en niveaux de gris** — réduction de 3 canaux BGR à 1 canal luminance
- **Normalisation du contraste** — égalisation d'histogramme pour corriger les variations d'éclairage
- **Binarisation (Otsu)** — conversion en noir et blanc pur avec seuillage automatique
- **Redressement (Deskewing)** — détection et correction de l'angle d'inclinaison

**Fonction principale :**
```python
preprocess(image_path: str) -> np.ndarray
```

---

### `src/ocr_header.py`
Extrait les métadonnées manuscrites de l'en-tête de la feuille :
- Découpage de la zone d'en-tête par ratio de hauteur
- Reconnaissance de texte via **PaddleOCR** (modèle layout-aware, langue française)
- Analyse par règles pour identifier les champs : module, enseignant, date, horaire, type de séance

**Fonction principale :**
```python
extract_header(binary_image: np.ndarray, header_ratio: float = 0.22) -> dict
```

**Exemple de sortie :**
```json
{
  "module": "Bases de Données",
  "teacher": "Pr. Hamim Mohammed",
  "date": "12/06/2025",
  "time": "10h00",
  "session_type": "TD"
}
```

---

### `src/checkbox_detection.py`
Détecte les absences dans le tableau étudiant :
- Découpage de la zone tableau (sous l'en-tête)
- Détection des lignes horizontales par morphologie mathématique (OpenCV)
- Segmentation du tableau en lignes individuelles
- Analyse de la **densité de pixels noirs** dans la colonne de signature de chaque ligne
- Classification : densité > seuil → présent ; densité ≤ seuil → absent

**Fonction principale :**
```python
detect_absences(binary_image: np.ndarray, header_ratio: float = 0.22,
                col_ratio: float = 0.85, density_threshold: float = 0.02) -> list[int]
```

---

### `src/name_matcher.py`
Fait la correspondance entre les indices de lignes absentes et les noms officiels des étudiants :
- Chargement de la base de données étudiants depuis `students.csv`
- Correspondance directe par index (feuille pré-imprimée = ordre fixe)
- Correction des erreurs OCR éventuelles via **distance de Levenshtein** (rapidfuzz)

**Fonction principale :**
```python
match_names(absent_indices: list[int], student_db_path: str) -> list[str]
```

---

### `src/pipeline.py`
Orchestre les quatre modules dans l'ordre et retourne le résultat final structuré.

**Fonction principale :**
```python
run(image_path: str, student_db_path: str,
    header_ratio: float = 0.22,
    col_ratio: float = 0.85,
    density_threshold: float = 0.02) -> dict
```

**Exemple de sortie :**
```json
{
  "image_path": "data/raw/sheet_001.jpg",
  "header_fields": {
    "module": "Bases de Données",
    "teacher": "Pr. Hamim Mohammed",
    "date": "12/06/2025",
    "time": "10h00",
    "session_type": "TD"
  },
  "absent_students": ["CHERKAOUI Fatima", "KADIRI Youssef"],
  "absent_indices": [1, 3]
}
```

---

### `app.py`
Interface web de démonstration construite avec **Streamlit** :
- Upload d'une feuille d'absence (JPG/PNG)
- Upload de la liste étudiants (CSV)
- Paramètres ajustables via la barre latérale (ratio en-tête, position colonne signature, seuil densité)
- Affichage structuré des métadonnées extraites et des absences détectées
- Export du résultat en **CSV** ou **JSON**

---

## Dépendances

| Bibliothèque | Version | Rôle |
|---|---|---|
| opencv-python | 4.x | Traitement d'image, morphologie, transformations affines |
| paddleocr | 3.x | OCR layout-aware, reconnaissance de l'écriture manuscrite |
| paddlepaddle | 3.x | Backend moteur de PaddleOCR |
| streamlit | 1.x | Interface web Python-only |
| rapidfuzz | 3.x | Distance de Levenshtein, correspondance floue de chaînes |
| pandas | 2.x | Manipulation de données, export CSV |
| numpy | 1.x | Calcul numérique, manipulation des tableaux image |

---

## Choix Technologiques

**PaddleOCR** a été retenu face à Tesseract pour sa gestion native de la mise en page (layout-aware OCR) : il détecte et segmente automatiquement les zones de texte avant reconnaissance, ce qui est déterminant pour un document structuré comme une feuille d'absence. Ses performances sur l'écriture manuscrite en français sont supérieures à Tesseract sans configuration avancée.

**La détection par densité de pixels** a été privilégiée face à un modèle de détection d'objets pour sa simplicité d'implémentation, son absence de besoin en données d'entraînement, et ses performances suffisantes dans le cadre d'un Proof of Concept.

**rapidfuzz** est utilisé à la place de python-Levenshtein pour ses performances (10 à 100 fois plus rapide) et son API `process.extractOne` qui gère la correspondance par lot en un seul appel.

---
