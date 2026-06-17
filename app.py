import streamlit as st
import os
import json
import pandas as pd
from PIL import Image
import re
import sys
import subprocess

# ── Resolve every path relative to this script so Streamlit's CWD doesn't matter ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def abspath(*parts):
    """Join parts and make them absolute relative to the project root."""
    return os.path.join(BASE_DIR, *parts)

# Ensure src modules are discoverable
sys.path.append(abspath("src"))
from name_matcher import map_filiere_to_csv

st.set_page_config(page_title="Gestion des Absences", layout="wide")

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def _get_str(series_row, col):
    """Safe column read from a pandas Series row."""
    return str(series_row[col]) if col in series_row.index and pd.notna(series_row[col]) else ""


def save_verified_data(doc_id, filiere, annee, seances_meta, extra_meta, edited_df, student_csv):
    """
    Save verified absence data into verified_output/<filename>.json.

    Parameters
    ----------
    doc_id       : str   – e.g. "doc_1"
    filiere      : str   – plain-text filière name
    annee        : str   – plain-text academic year
    seances_meta : dict  – the seances sub-dict (already plain strings, not OCR confidence objects)
    extra_meta   : dict  – remaining metadata fields (enseignant, module, …)
    edited_df    : DataFrame – the edited attendance table from st.data_editor
    student_csv  : str   – csv filename used for matching
    """
    out_dir = abspath("verified_output")
    os.makedirs(out_dir, exist_ok=True)

    safe_filiere = re.sub(r'[^a-zA-Z0-9_\-]', '_', filiere.strip()) or "Inconnue"
    safe_annee   = re.sub(r'[^a-zA-Z0-9_\-]', '_', annee.strip())   or "Inconnue"
    final_json_path = os.path.join(out_dir, f"Absences_{safe_filiere}_{safe_annee}.json")

    # Load existing file or start fresh
    if os.path.exists(final_json_path):
        with open(final_json_path, 'r', encoding='utf-8') as f:
            global_data = json.load(f)
    else:
        global_data = {"filiere": filiere.strip(), "annee": annee.strip(), "documents": {}}

    # Build absence list from the edited dataframe
    absences_list = []
    for _, row in edited_df.iterrows():
        student_data = {
            "n_apo":   _get_str(row, "N° Apo"),
            "nom":     _get_str(row, "Nom"),
            "prenom":  _get_str(row, "Prénom"),
        }
        for i in range(1, 11):
            col_name = f"S{i}"
            if col_name in row.index:
                student_data[f"seance_{i}"] = {"is_present": bool(row[col_name])}
        absences_list.append(student_data)

    # Build the full metadata block to store
    metadata_block = {
        "filiere":    filiere.strip(),
        "annee":      annee.strip(),
        "enseignant": extra_meta.get("enseignant", ""),
        "module":     extra_meta.get("module", ""),
        "seances":    seances_meta,
    }

    global_data["documents"][doc_id] = {
        "metadata":        metadata_block,
        "student_list_csv": student_csv,
        "absences":        absences_list,
    }

    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(global_data, f, indent=4, ensure_ascii=False)

    return final_json_path


def main():
    st.title("Système de Gestion des Absences")
    st.markdown("---")

    # ── Directories (always absolute) ──
    raw_dir = abspath("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(abspath("data", "preprocessed"), exist_ok=True)
    os.makedirs(abspath("data", "cropped"), exist_ok=True)
    os.makedirs(abspath("data", "rows"), exist_ok=True)
    os.makedirs(abspath("data", "output"), exist_ok=True)
    os.makedirs(abspath("debug"), exist_ok=True)
    os.makedirs(abspath("verified_output"), exist_ok=True)
    os.makedirs(abspath("config", "groups"), exist_ok=True)

    # ── Session State ──
    if "sidebar_selected_doc" not in st.session_state:
        st.session_state.sidebar_selected_doc = None

    doc_ids = sorted(
        [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))],
        key=natural_sort_key,
    )

    if not doc_ids:
        st.sidebar.warning("Aucun document trouvé. Veuillez en uploader dans l'onglet 'Docs Bruts'.")
        selected_doc = None
    else:
        st.sidebar.header("Configuration")

        index = 0
        if st.session_state.sidebar_selected_doc in doc_ids:
            index = doc_ids.index(st.session_state.sidebar_selected_doc)

        selected_doc = st.sidebar.selectbox("Document Actuel :", doc_ids, index=index, key="sb_doc_select")
        st.session_state.sidebar_selected_doc = selected_doc

        # Preview doc data for CSV auto-selection
        doc_data_preview = {}
        for try_path in [
            abspath("data", "output", f"{selected_doc}.json"),
            abspath("data", "output", selected_doc, "metadata.json"),
        ]:
            if os.path.exists(try_path):
                try:
                    with open(try_path, 'r', encoding='utf-8') as f:
                        doc_data_preview = json.load(f)
                    break
                except Exception:
                    pass

        config_dir = abspath("config", "groups")
        csv_files = [f for f in os.listdir(config_dir) if f.endswith('.csv')] if os.path.exists(config_dir) else []
        default_csv_index = 0
        if csv_files and "filiere" in doc_data_preview:
            filiere_name = doc_data_preview.get("filiere", {})
            filiere_name = filiere_name.get("value", "") if isinstance(filiere_name, dict) else str(filiere_name)
            if filiere_name:
                mapped_csv = map_filiere_to_csv(filiere_name)
                if mapped_csv in csv_files:
                    default_csv_index = csv_files.index(mapped_csv)

        if csv_files:
            selected_csv = st.sidebar.selectbox(
                "Base de données Étudiants (CSV) :", csv_files, index=default_csv_index
            )
            if default_csv_index != 0:
                st.sidebar.caption("CSV auto-sélectionné par l'OCR.")
        else:
            st.sidebar.warning("Aucun CSV d'étudiants trouvé.")
            selected_csv = None

        st.sidebar.markdown("---")
        st.sidebar.subheader("Ajouter une Base (CSV)")
        uploaded_csv = st.sidebar.file_uploader("Upload un nouveau CSV", type=['csv'])
        if uploaded_csv:
            with open(os.path.join(config_dir, uploaded_csv.name), "wb") as f:
                f.write(uploaded_csv.getbuffer())
            st.sidebar.success(f"{uploaded_csv.name} ajouté !")
            st.rerun()

    # ── TABS ──
    tab_docs, tab_orch, tab_meta, tab_abs, tab_json = st.tabs([
        "1. Docs Bruts",
        "2. Orchestrateur",
        "3. Métadonnées",
        "4. Absences & Vérification",
        "5. Visionneuse JSON",
    ])

    # ── TAB 1: GESTION DES DOCUMENTS ──
    with tab_docs:
        st.subheader("Uploader de nouvelles images")
        col_new1, col_new2 = st.columns([1, 2])

        highest_id = 0
        for d in doc_ids:
            m = re.search(r'^doc_(\d+)$', d)
            if m:
                highest_id = max(highest_id, int(m.group(1)))
        next_id = f"doc_{highest_id + 1}"

        with col_new1:
            new_doc_id = st.text_input("ID du Document (Automatique)", value=next_id)
        with col_new2:
            uploaded_files = st.file_uploader(
                "Sélectionnez les feuilles d'absences", accept_multiple_files=True, type=['png', 'jpg', 'jpeg']
            )

        if st.button("Sauvegarder le Document", type="primary"):
            if not new_doc_id.startswith("doc_"):
                st.error("L'ID du document doit commencer par 'doc_'")
            elif uploaded_files:
                doc_path = os.path.join(raw_dir, new_doc_id)
                os.makedirs(doc_path, exist_ok=True)
                for uf in uploaded_files:
                    with open(os.path.join(doc_path, uf.name), "wb") as f:
                        f.write(uf.getbuffer())
                st.session_state.sidebar_selected_doc = new_doc_id
                st.success(f"Document '{new_doc_id}' créé avec succès !")
                st.rerun()
            else:
                st.warning("Veuillez uploader au moins une image.")

        st.markdown("---")
        st.subheader("Gérer les images d'un document existant")
        if doc_ids:
            manage_doc = st.selectbox("Sélectionnez le document à modifier :", doc_ids, key="manage_docs")
            doc_path = os.path.join(raw_dir, manage_doc)
            images = sorted([f for f in os.listdir(doc_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if images:
                cols = st.columns(4)
                for idx, img_file in enumerate(images):
                    img_p = os.path.join(doc_path, img_file)
                    with cols[idx % 4]:
                        st.image(Image.open(img_p), caption=img_file, use_container_width=True)
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("Pivoter 90°", key=f"rot_{manage_doc}_{img_file}"):
                                with Image.open(img_p) as im:
                                    im.rotate(-90, expand=True).save(img_p)
                                st.rerun()
                        with c2:
                            if st.button("Supprimer", key=f"del_{manage_doc}_{img_file}"):
                                os.remove(img_p)
                                st.rerun()
            else:
                st.info("Aucune image dans ce document.")
                if st.button("Supprimer le dossier vide", type="primary"):
                    os.rmdir(doc_path)
                    if st.session_state.sidebar_selected_doc == manage_doc:
                        st.session_state.sidebar_selected_doc = None
                    st.rerun()

    if not selected_doc:
        st.stop()

    # ── TAB 2: ORCHESTRATEUR DE PIPELINE ──
    with tab_orch:
        st.subheader("Orchestrateur & Débogage du Pipeline")

        if st.button("🚀 Lancer le Pipeline Complet (Mode Auto)", type="primary", use_container_width=True):
            with st.spinner("Exécution de toutes les étapes... Cela peut prendre plusieurs minutes."):
                r1     = subprocess.run(["python", abspath("src", "preprocessing.py"), selected_doc], capture_output=True, text=True, cwd=BASE_DIR)
                r2     = subprocess.run(["python", abspath("src", "cropper.py"),       selected_doc], capture_output=True, text=True, cwd=BASE_DIR)
                r3     = subprocess.run(["python", abspath("src", "row_slicer.py"),    selected_doc], capture_output=True, text=True, cwd=BASE_DIR)
                r_cell = subprocess.run(["python", abspath("src", "cell_slicer.py"),   selected_doc], capture_output=True, text=True, cwd=BASE_DIR)
                r4     = subprocess.run(["python", abspath("src", "ocr_header.py"),    selected_doc], capture_output=True, text=True, cwd=BASE_DIR)

                std_args = ["python", abspath("src", "ocr_students.py"), selected_doc]
                if selected_csv:
                    std_args.extend(["--csv", selected_csv])
                r5 = subprocess.run(std_args, capture_output=True, text=True, cwd=BASE_DIR)

                if all(r.returncode == 0 for r in [r1, r2, r3, r_cell, r4, r5]):
                    st.success("Pipeline complet terminé avec succès !")
                    st.rerun()
                else:
                    st.error("Erreur durant l'exécution. Vérifiez les fichiers.")
                    for r, name in [(r1,"preprocessing"),(r2,"cropper"),(r3,"row_slicer"),(r_cell,"cell_slicer"),(r4,"ocr_header"),(r5,"ocr_students")]:
                        if r.returncode != 0:
                            st.error(f"[{name}] {r.stderr}")

        st.markdown("---")

        def show_gallery(directory, width=200, limit=20, unmask=False):
            if os.path.exists(directory):
                imgs = sorted(
                    [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png'))],
                    key=natural_sort_key,
                )
                if imgs:
                    lim = len(imgs) if unmask else limit
                    cols = st.columns(min(len(imgs[:lim]), 4))
                    for i, img in enumerate(imgs[:lim]):
                        with cols[i % 4]:
                            st.image(os.path.join(directory, img), use_container_width=True, caption=img)
                    if len(imgs) > lim:
                        st.caption(f"... et {len(imgs)-lim} autres images masquées.")
                else:
                    st.info("Dossier vide.")
            else:
                st.info("Dossier non généré.")

        st.markdown("#### Étape 1 : Nettoyage (Preprocessing)")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("Exécuter Preprocessing"):
                res = subprocess.run(["python", abspath("src", "preprocessing.py"), selected_doc], capture_output=True, text=True, cwd=BASE_DIR)
                if res.returncode == 0: st.rerun()
                else: st.error(res.stderr)
        with c2:
            show_gallery(abspath("data", "preprocessed", selected_doc), width=150)

        st.markdown("#### Étape 2 : Découpage (Cropper)")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("Exécuter Découpage"):
                res = subprocess.run(["python", abspath("src", "cropper.py"), selected_doc], capture_output=True, text=True, cwd=BASE_DIR)
                if res.returncode == 0: st.rerun()
                else: st.error(res.stderr)
        with c2:
            show_gallery(abspath("data", "cropped", selected_doc), width=250)

        st.markdown("#### Étape 3 : Isolation des Étudiants (Row Slicer)")
        unmask_rows = st.checkbox("Afficher toutes les lignes (Démasquer)")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("Exécuter Lignes"):
                res = subprocess.run(["python", abspath("src", "row_slicer.py"), selected_doc], capture_output=True, text=True, cwd=BASE_DIR)
                if res.returncode == 0: st.rerun()
                else: st.error(res.stderr)
        with c2:
            show_gallery(abspath("data", "rows", selected_doc), width=350, limit=4, unmask=unmask_rows)

        st.markdown("#### Étape 3.5 : Isolation des Cellules (Cell Slicer)")
        unmask_cells = st.checkbox("Afficher la première cellule de toutes les lignes")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("Exécuter Cellules"):
                res = subprocess.run(["python", abspath("src", "cell_slicer.py"), selected_doc], capture_output=True, text=True, cwd=BASE_DIR)
                if res.returncode == 0: st.rerun()
                else: st.error(res.stderr)
        with c2:
            cells_dir = abspath("data", "cells", selected_doc)
            if os.path.exists(cells_dir):
                cell_preview = []
                for row_d in sorted(os.listdir(cells_dir), key=natural_sort_key):
                    p = os.path.join(cells_dir, row_d, "seance_1.jpg")
                    if os.path.exists(p):
                        cell_preview.append(p)
                if cell_preview:
                    lim = len(cell_preview) if unmask_cells else 6
                    cols = st.columns(min(len(cell_preview[:lim]), 6))
                    for i, img in enumerate(cell_preview[:lim]):
                        with cols[i % 6]:
                            st.image(img, use_container_width=True, caption=os.path.basename(os.path.dirname(img)))
                else:
                    st.info("Aucune cellule trouvée.")
            else:
                st.info("Dossier non généré.")

        st.markdown("#### Étape 4 : Extraction En-tête (OCR)")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("Exécuter OCR En-tête"):
                res = subprocess.run(["python", abspath("src", "ocr_header.py"), selected_doc], capture_output=True, text=True, cwd=BASE_DIR)
                if res.returncode == 0: st.rerun()
                else: st.error(res.stderr)
        with c2:
            p = abspath("data", "output", selected_doc, "metadata.json")
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    st.json(json.load(f), expanded=False)
            else:
                st.info("JSON non généré.")

        st.markdown("#### Étape 5 : Extraction Absences (OCR)")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("Exécuter OCR Étudiants"):
                std_args = ["python", abspath("src", "ocr_students.py"), selected_doc]
                if selected_csv:
                    std_args.extend(["--csv", selected_csv])
                res = subprocess.run(std_args, capture_output=True, text=True, cwd=BASE_DIR)
                if res.returncode == 0: st.rerun()
                else: st.error(res.stderr)
        with c2:
            p = abspath("data", "output", selected_doc, "absences.json")
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    st.json(json.load(f), expanded=False)
            else:
                st.info("JSON non généré.")

    # ── LOAD DOC DATA (always absolute paths) ──
    doc_data = {}
    p1 = abspath("data", "output", f"{selected_doc}.json")
    p2 = abspath("data", "output", selected_doc, "metadata.json")
    p3 = abspath("data", "output", selected_doc, "absences.json")

    if os.path.exists(p1):
        with open(p1, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
    elif os.path.exists(p2) and os.path.exists(p3):
        with open(p2, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        with open(p3, 'r', encoding='utf-8') as f:
            doc_data["absences"] = json.load(f)

    # ── Extract filiere/annee ONCE and cache in session_state ──
    # This ensures they are available regardless of which tab is active when saving.
    def _extract_plain(doc_data, key):
        val = doc_data.get(key, "")
        if isinstance(val, dict):
            return val.get("value", "")
        return str(val) if val else ""

    if doc_data:
        raw_filiere = _extract_plain(doc_data, "filiere")
        raw_annee   = _extract_plain(doc_data, "annee")
        # Store in session_state keyed to this doc so switching docs updates them
        st.session_state[f"filiere_{selected_doc}"] = raw_filiere
        st.session_state[f"annee_{selected_doc}"]   = raw_annee

    # ── TAB 3: METADATA REVIEW ──
    # Collect user edits into session_state keys so Tab 4 can read them regardless of tab order
    with tab_meta:
        if not doc_data:
            st.info("Lancez d'abord l'OCR En-tête dans l'Orchestrateur.")
        else:
            st.subheader("Informations de l'En-tête")

            meta_keys = ["filiere", "annee", "enseignant", "module"]
            col1, col2 = st.columns(2)
            for i, key in enumerate(meta_keys):
                target_col = col1 if i % 2 == 0 else col2
                val_obj = doc_data.get(key, {"value": "", "confidence": 0.0})
                val = val_obj.get("value", "") if isinstance(val_obj, dict) else str(val_obj)
                display_key = "Filière" if key == "filiere" else "Année" if key == "annee" else key.capitalize()
                with target_col:
                    edited_val = st.text_input(display_key, value=val, key=f"meta_{key}")
                    # Cache every edit into session_state so other tabs can use them
                    st.session_state[f"{key}_{selected_doc}"] = edited_val

            st.markdown("#### Séances Détectées")
            seances_data = doc_data.get("seances", {})
            cols_s = st.columns(5)
            for i in range(1, 11):
                s_key  = f"seance{i}"
                s_info = seances_data.get(s_key, {})
                with cols_s[(i - 1) % 5]:
                    st.markdown(f"**Séance {i}**")
                    date_v  = s_info.get("date",        {}).get("value", "") if isinstance(s_info.get("date"),        dict) else ""
                    hdeb_v  = s_info.get("heure_debut", {}).get("value", "") if isinstance(s_info.get("heure_debut"), dict) else ""
                    hfin_v  = s_info.get("heure_fin",   {}).get("value", "") if isinstance(s_info.get("heure_fin"),   dict) else ""
                    typ_v   = s_info.get("type",        {}).get("value", "") if isinstance(s_info.get("type"),        dict) else ""

                    st.session_state[f"s{i}_date_{selected_doc}"]  = st.text_input(f"Date S{i}",  value=date_v,  key=f"d_{i}")
                    st.session_state[f"s{i}_hdeb_{selected_doc}"]  = st.text_input(f"Début S{i}", value=hdeb_v,  key=f"hd_{i}")
                    st.session_state[f"s{i}_hfin_{selected_doc}"]  = st.text_input(f"Fin S{i}",   value=hfin_v,  key=f"hf_{i}")
                    st.session_state[f"s{i}_type_{selected_doc}"]  = st.text_input(f"Type S{i}",  value=typ_v,   key=f"t_{i}")

            # Show physical header image for verification
            st.markdown("---")
            st.subheader("Image de l'En-tête (Pour vérification)")
            crop_dir = abspath("data", "cropped", selected_doc)
            if os.path.exists(crop_dir):
                headers = [f for f in os.listdir(crop_dir) if "header" in f.lower()]
                if headers:
                    st.image(os.path.join(crop_dir, headers[0]), use_container_width=True)
                else:
                    st.info("Aucune image d'en-tête générée.")
            else:
                st.info("Dossier de découpage introuvable.")

    # ── Helper: build metadata dicts from session_state (works from any tab) ──
    def get_current_filiere():
        return st.session_state.get(f"filiere_{selected_doc}", _extract_plain(doc_data, "filiere"))

    def get_current_annee():
        return st.session_state.get(f"annee_{selected_doc}", _extract_plain(doc_data, "annee"))

    def get_current_seances():
        seances = {}
        for i in range(1, 11):
            seances[f"seance{i}"] = {
                "date":        st.session_state.get(f"s{i}_date_{selected_doc}", ""),
                "heure_debut": st.session_state.get(f"s{i}_hdeb_{selected_doc}", ""),
                "heure_fin":   st.session_state.get(f"s{i}_hfin_{selected_doc}", ""),
                "type":        st.session_state.get(f"s{i}_type_{selected_doc}", ""),
            }
        return seances

    def get_extra_meta():
        return {
            "enseignant": st.session_state.get(f"enseignant_{selected_doc}", _extract_plain(doc_data, "enseignant")),
            "module":     st.session_state.get(f"module_{selected_doc}",     _extract_plain(doc_data, "module")),
        }

    # ── TAB 4: ABSENCES & VERIFICATION ──
    with tab_abs:
        if not doc_data or "absences" not in doc_data:
            st.info("Lancez d'abord l'OCR Étudiants dans l'Orchestrateur.")
        else:
            col_table, col_img = st.columns([2, 1])

            with col_table:
                st.subheader("Liste de Présence")
                st.caption("Cochez ✅ = présent, décochez = absent. Cliquer enregistre la signature à droite.")
                absences = doc_data.get("absences", [])

                df_data = []
                for idx, r in enumerate(absences):
                    row = {
                        "_internal_idx": idx,
                        "Row ID": r.get('row_index', ''),
                        "N° Apo":  r.get('n_apo', ''),
                        "Nom":     r.get('nom', ''),
                        "Prénom":  r.get('prenom', ''),
                        "Confiance (%)": r.get('match_confidence', 0.0),
                    }
                    sessions = r.get('sessions', {})
                    for i in range(1, 11):
                        s_data = sessions.get(f"seance{i}", {})
                        row[f"S{i}"] = bool(s_data.get("is_present", False))
                    df_data.append(row)

                df = pd.DataFrame(df_data)
                df.sort_values(
                    by="N° Apo",
                    key=lambda col: pd.to_numeric(col, errors='coerce'),
                    inplace=True,
                )
                df.reset_index(drop=True, inplace=True)

                column_config = {
                    "_internal_idx":  None,
                    "Row ID":         None,
                    "N° Apo":         st.column_config.TextColumn(disabled=True),
                    "Nom":            st.column_config.TextColumn(disabled=True),
                    "Prénom":         st.column_config.TextColumn(disabled=True),
                    "Confiance (%)":  st.column_config.NumberColumn(disabled=True, format="%.1f"),
                }
                for i in range(1, 11):
                    column_config[f"S{i}"] = st.column_config.CheckboxColumn(label=f"S{i}")

                # Initialise baseline only once per document
                if f"prev_df_{selected_doc}" not in st.session_state:
                    st.session_state[f"prev_df_{selected_doc}"] = df.copy()

                edited_df = st.data_editor(
                    df,
                    column_config=column_config,
                    hide_index=True,
                    use_container_width=True,
                    num_rows="fixed",
                    key="attendance_editor",
                )

                # Detect which cell changed → update image viewer
                prev_df = st.session_state[f"prev_df_{selected_doc}"]
                try:
                    diff = edited_df.ne(prev_df)
                    changed_rows = diff.any(axis=1)
                    if changed_rows.any():
                        changed_idx = changed_rows[changed_rows].index[0]
                        st.session_state.active_physical_row = df.loc[changed_idx, "Row ID"]
                        row_diff     = diff.loc[changed_idx]
                        changed_cols = row_diff[row_diff].index.tolist()
                        if changed_cols:
                            changed_col = changed_cols[0]
                            if changed_col.startswith("S"):
                                st.session_state.active_seance = changed_col[1:]  # strip "S"
                        st.session_state[f"prev_df_{selected_doc}"] = edited_df.copy()
                except Exception:
                    # Shape mismatch guard (e.g. first render)
                    st.session_state[f"prev_df_{selected_doc}"] = edited_df.copy()

                if st.button("💾 Enregistrer les données", type="primary"):
                    student_csv = doc_data.get("student_list_csv", selected_csv)
                    final_path  = save_verified_data(
                        doc_id      = selected_doc,
                        filiere     = get_current_filiere(),
                        annee       = get_current_annee(),
                        seances_meta= get_current_seances(),
                        extra_meta  = get_extra_meta(),
                        edited_df   = edited_df,
                        student_csv = student_csv,
                    )
                    st.success(f"✅ Sauvegardé dans : `{final_path}`")
                    # Reset baseline so subsequent edits diff correctly
                    st.session_state.pop(f"prev_df_{selected_doc}", None)

            with col_img:
                st.subheader("Vérification Visuelle")

                active_row = st.session_state.get("active_physical_row", None)
                if active_row is not None and active_row != "" and pd.notna(active_row):
                    physical_row_index = int(active_row)
                    seance_id = st.session_state.get("active_seance", "1")

                    cell_img_path = abspath("data", "cells", selected_doc, f"row_{physical_row_index:03d}", f"seance_{seance_id}.jpg")
                    row_img_path  = abspath("data", "rows",  selected_doc, f"row_{physical_row_index:03d}.jpg")

                    if os.path.exists(cell_img_path):
                        st.image(Image.open(cell_img_path), width=200, caption=f"Étudiant n°{physical_row_index} – Séance {seance_id}")
                    elif os.path.exists(row_img_path):
                        st.image(Image.open(row_img_path), use_container_width=True, caption=f"Ligne n°{physical_row_index}")
                        st.warning("Cellule individuelle introuvable. Exécutez 'Isolation des Cellules' dans l'Orchestrateur.")
                    else:
                        st.warning("Image de la cellule et de la ligne introuvables.")
                else:
                    st.info("👈 Cochez ou décochez une case pour voir la signature correspondante.")

    # ── TAB 5: JSON Viewer ──
    with tab_json:
        st.subheader("Visionneuse JSON & Export")

        json_choice = st.radio("Sélectionnez le fichier à inspecter :", [
            "1. Sortie OCR En-tête (metadata.json)",
            "2. Sortie OCR Étudiants (absences.json)",
            "3. Base de données finale (verified_output/)",
        ])

        target_path = None
        if "metadata.json" in json_choice:
            target_path = abspath("data", "output", selected_doc, "metadata.json")
        elif "absences.json" in json_choice:
            target_path = abspath("data", "output", selected_doc, "absences.json")
        elif "Base de données finale" in json_choice:
            # Use session_state values (populated from tab_meta edits) with raw OCR fallback
            filiere = get_current_filiere()
            annee   = get_current_annee()
            sf = re.sub(r'[^a-zA-Z0-9_\-]', '_', filiere) or "Inconnue"
            sa = re.sub(r'[^a-zA-Z0-9_\-]', '_', annee)   or "Inconnue"
            target_path = abspath("verified_output", f"Absences_{sf}_{sa}.json")

        if target_path and os.path.exists(target_path):
            with open(target_path, 'r', encoding='utf-8') as f:
                json_content = f.read()

            col_d1, col_d2 = st.columns([1, 4])
            with col_d1:
                st.download_button(
                    label="⬇ Exporter le JSON",
                    data=json_content,
                    file_name=os.path.basename(target_path),
                    mime="application/json",
                    type="primary",
                )
            with col_d2:
                st.caption(f"Fichier local : `{target_path}`")

            st.json(json.loads(json_content))
        else:
            if "Base de données finale" in json_choice:
                st.info(
                    "Fichier non trouvé. Allez dans l'onglet **Absences & Vérification**, "
                    "vérifiez les données, puis cliquez sur **💾 Enregistrer les données**."
                )
            else:
                st.info("Fichier non trouvé. Lancez d'abord l'étape OCR correspondante.")


if __name__ == "__main__":
    main()
