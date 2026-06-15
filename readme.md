# Pipeline d'Automatisation des Feuilles d'Absence

![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-lightgrey?logo=tesseract&logoColor=black)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)

**Extraction automatique des métadonnées et des absences d'étudiants à partir de feuilles de preésence scannées, en combinant Vision par Ordinateur, Reconnaissance d'Ecriture Mqnuscrite (HTR) et correspondance floue.**

---

## Table des matières

- [Pipeline d'Automatisation des Feuilles d'Absence](#pipeline-dautomatisation-des-feuilles-dabsence)
  - [Table des matières](#table-des-matières)
  - [Contexte et problematique](#contexte-et-problematique)
  - [Fonctionnalites](#fonctionnalites)
  - [Technologies utilisees](#technologies-utilisees)
  - [Structure du projet](#structure-du-projet)
  - [Instalation et configuration](#instalation-et-configuration)
  - [Configuration des dictionnaires](#configuration-des-dictionnaires)
- [Utilisation](#utilisation)
  - [Pipeline en ligne de commande](#pipeline-en-ligne-de-commande)
  - [Démonstration Streamlit](#démonstration-streamlit)
- [Vue d’ensemble des modules](#vue-densemble-des-modules)

---

## Contexte et problematique 

A l'Ecole Supérieure de Technologie de Fkih ben Salah (Maroc), la gestion des absences repose encore sur des feuilles pré-imprimées remplies manuellement par les enseignants . Ceux-ci y inscrivent à la main les métadonnées du cours (module, date, heure, type) et peuvent émarger les étudiants (signature, "P" ou "Abs"). La ressaisie dans le système d'information est chronophage et source d'erreurs.

Ce projet vise à **automatiser l'extraction** de ces informations à partir de scans, en générant des fichiers structurés (JSON, CSV) prêts à intégrer le SI de l'école.

> Ce projet est un **Proof of Concept** réalisé dans le cadre d'un stage interne de l'EST Fkih Ben Salah. L'architecture du pipeline est solide, mais les modules de reconnaissance d'écriture et de détection d'abscences ne sont **pas assez fiables pour un usage en production** (cf. [Limitations](#limitations)).

---

## Fonctionnalites

- **Prétraitement d'image** : redressement, amélioration du contraste (CLAHE), recadrage automatique sur la grille du tableau.
- **Séparation en-tête / tableau** : détection robuste de la ligne contenant les entêtes de colonnes ("N° Apo", "Nom", "Prénom").
- **Reconnaissqnce de l'écriture ,manuscrite (HTR)** : utilisation de TrOCR (Microsoft) pour lire les champs manuscrits (enseignant, module, date, heure, type de seance), normalisées ensuite par correspondance floue sur des dictionnaires configurables.
- **Identification des étudiants** : OCR (Tesseract) sur les noms et numéros Apogee imprimes, mise en correspondance avec des listes officielles (CSV) via distance de Levenshtein et ratio de similarité tokenisé.
- **Detection des absences** : heuristique hybride combinant densité de pixels et mini-OCR ciblé ('Abs', "Present", ou signature d'un étudiant) dans chaque cellule d'emargement.
- **Sorties structurees** : fichiers JSON (métadonnées + liste d'absences) et export CSV.
- **Interfaces Streamlit** : téléversement d'un scan, lancement du pipeline, visualisation interactive et téléchargement des résultats.

---

## Technologies utilisees

| Domaine                  | Outil / Modèle                                         |
| ------------------------ | ------------------------------------------------------ |
| Traitement d’image       | OpenCV, NumPy                                          |
| OCR imprimé              | Tesseract (via `pytesseract`)                          |
| OCR manuscrit            | TrOCR-large-handwritten (Hugging Face Transformers)    |
| Correspondance floue     | RapidFuzz                                              |
| Manipulation de données  | Pandas, PyYAML                                         |
| Interface Web            | Streamlit                                              |

---

## Structure du projet
```
absence-pipeline/
├── app.py # Application Streamlit
├── requirements.txt # Dépendances Python
├── config/
│ ├── metadata_dict.yaml # Dictionnaires de métadata (enseignants, modules, filières, horaires...)
│ ├── filiere_mapping.json # Dictionnaires d'Association filière - fichier CSV
│ ├── absence_output_format.json
│ ├── metadata_output_format.json
│ └── groups/ # Listes officielle d’étudiants par classe (CSV)
├── data/
│ ├── raw/ # Scans originaux organisés par document (doc_1/as1.jpg ...)
│ ├── preprocessed/ # Images prétraitées
│ ├── cropped/ # Images en-tête et tableau séparées
│ └── output/ # Fichiers JSON (pre-verification)
├── src/
│ ├── preprocessing.py # Étape 1 : redressement, CLAHE, recadrage
│ ├── cropper.py # Étape 2 : séparation en-tête / tableau
│ ├── ocr_header.py # Étape 3 : extraction des métadonnées (TrOCR + fuzzy)
│ ├── row_slicer.py # Étape 4 : découpage du tableau en lignes
│ ├── ocr_students.py # Étape 5 : OCR noms, détection absences, matching
│ ├── name_matcher.py # Logique de correspondance floue
│ ├── expand_dict.py # Génération de motifs regex pour les dictionnaires
│ ├── table_analyzer.py # Analyse alternative du tableau (obsolète)
│ └── checkbox_detection.py# Détection alternative par densité
|
└── verified_output/ 
```
---

## Instalation et configuration

1. **Cloner le dépôt**
```bash
git clone https://github.com/votre-nom/absence-pipeline.git
cd absence-pipeline
```

1. **Créer un environnement virtuel** (Python 3.12 recommandé)
```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

1. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Installer Tesseract OCR**
  - Télécharger depuis [https://tesseractocr.org/](https://tesseractocr.org/) et s’assurer que tesseract.exe est dans le PATH, ou ajuster le chemin dans les scripts src/ocr_header.py et src/ocr_students.py.

5. Téléchargement du modèle TrOCR
  - Le premier lancement de ocr_header.py téléchargera automatiquement microsoft/trocr-large-handwritten (≈2 Go). Une bonne connexion et un espace disque suffisant sont nécessaires.

## Configuration des dictionnaires

Avant d’utiliser le pipeline sur vos propres documents, adaptez les fichiers de configuration :

- ```config/metadata_dict.yaml```
    Contient toutes les valeurs possibles pour les clés `filieres`, `enseignants`, `modules`, `heure_debut`, `heure_fin`, `types`, `annee`. Ajoutez-y les libellés approprie a votre utilisation.
- ```config/filiere_mapping.json```
    Fait le lien entre le texte reconnu par OCR pour la filière et le fichier CSV correspondant dans `config/groups/`. Exemple :
    ```json
    {
    "Big Data": "big_data.csv",
    "Infrastructure, traitement et annalyse des donnees massives": "big_data.csv",
    "Génie Informatique 1ère année": "Genie_Informatique_1ere_annee.csv"
    }
    ```
- ```config/groups/*.csv```
    Listes officielles d’étudiants, avec les colonnes n_apo, nom, prenom. L’absence de n_apo dégrade la qualité du matching.

---

# Utilisation

Tous les modules se lancent depuis la racine du projet.

## Pipeline en ligne de commande

1. Placer les scans dans `data/raw/<doc_id>/` sous la forme `as1.jpg`, `as2.jpg`, etc.
Exemple : `data/raw/doc_1/as1.jpg, as2.jpg`

2. Prétraitement
    ```bash
    python src/preprocessing.py doc_1
    ```
    ➔ `as1.jpg`, `as2.jpg` ...etc pretraiter dans `data/preprocessed/`
3. Séparation en‑tête / tableau

   ```bash
   python src/cropper.py doc_1
   ```
    ➔ `as1_header.jpg` et `as1_table.jpg` dans d`ata/cropped/doc_1/`.

4. extraction des métadonnées

    ```bash
    python src/ocr_header.py doc_1
    ```
    ➔ `data/output/doc_1.json` (métadonnées).

5. Découpage du tableau en lignes

    ```bash
    python src/row_slicer.py doc_1
    ```
    ➔ Lignes individuelles dans `data/rows/doc_1/row_001`.jpg ...

6. OCR étudiants et détection d’absences

    ```bash
    python src/ocr_students.py doc_1
    ```
    ➔ Mise à jour de `data/output/doc_1.json` et création de absences.json et metadata.json dans data/output/doc_1/.

## Démonstration Streamlit

  ```bash
  streamlit run app.py
  ```

Téléversez un scan, selectioner la base de données appropriée (fichier csv), lancez le pipeline, révisez les données extraites et exportez en JSON.

# Vue d’ensemble des modules

```graph LR

    A[Image brute] --> B(preprocessing.py<br/>Redressement, CLAHE, recadrage)
    B --> C(cropper.py<br/>Séparation en-tête / tableau)
    C --> D(ocr_header.py<br/>TrOCR + fuzzy matching)
    C --> E(row_slicer.py<br/>Découpage en lignes)
    E --> F(ocr_students.py<br/>OCR noms + détection absences + matching DB)
    F --> G(JSON structuré)
    D --> G
    G --> H(app.py<br/>Streamlit demo)
    ```