import streamlit as st
import os
import json
import pandas as pd
from PIL import Image
import re
import sys
import subprocess

# Ensure src modules are discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from name_matcher import map_filiere_to_csv

st.set_page_config(page_title="Gestion des Absences", layout="wide")

def save_verified_data(doc_id, updated_meta, edited_df, student_csv):
    """
    Enregistre les données vérifiées dans un fichier JSON centralisé par filière et année.
    """
    out_dir = "verified_output"
    os.makedirs(out_dir, exist_ok=True)
    
    filiere = updated_meta.get("filiere", "Inconnue").strip()
    annee = updated_meta.get("annee", "Inconnue").strip()
    
    safe_filiere = re.sub(r'[^a-zA-Z0-9_\-]', '_', filiere)
    safe_annee = re.sub(r'[^a-zA-Z0-9_\-]', '_', annee)
    
    final_json_path = os.path.join(out_dir, f"Absences_{safe_filiere}_{safe_annee}.json")
    
    if os.path.exists(final_json_path):
        with open(final_json_path, 'r', encoding='utf-8') as f:
            global_data = json.load(f)
    else:
        global_data = {
            "filiere": filiere,
            "annee": annee,
            "documents": {}
        }
        
    absences_list = []
    for _, row in edited_df.iterrows():
        student_data = {
            "n_apo": row.get("N° Apo", ""),
            "nom": row.get("Nom", ""),
            "prenom": row.get("Prénom", ""),
        }
        for i in range(1, 11):
            col_name = f"Séance {i}"
            if col_name in row:
                student_data[f"seance_{i}"] = row[col_name]
        absences_list.append(student_data)
        
    global_data["documents"][doc_id] = {
        "metadata": updated_meta,
        "student_list_csv": student_csv,
        "absences": absences_list
    }
    
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(global_data, f, indent=4, ensure_ascii=False)
        
    return final_json_path


def main():
    st.title("Système de Gestion des Absences")
    st.markdown("---")

    # 1. Setup Directories
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(os.path.join("data", "preprocessed"), exist_ok=True)
    os.makedirs(os.path.join("data", "cropped"), exist_ok=True)
    os.makedirs(os.path.join("data", "rows"), exist_ok=True)
    os.makedirs(os.path.join("data", "output"), exist_ok=True)
    os.makedirs("debug", exist_ok=True)
    os.makedirs("verified_output", exist_ok=True)
    
    # -------------------
    # SIDEBAR SETUP
    # -------------------
    doc_ids = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
    if not doc_ids:
        st.sidebar.warning("Aucun document trouvé. Veuillez en uploader dans l'onglet 'Docs Bruts'.")
        selected_doc = None
    else:
        st.sidebar.header("Configuration")
        selected_doc = st.sidebar.selectbox("Document Actuel :", doc_ids)
        
        # Auto-select CSV logic
        json_path = os.path.join("data", "output", f"{selected_doc}.json")
        alt_json_path = os.path.join("data", "output", selected_doc, "metadata.json")
        doc_data_preview = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f: doc_data_preview = json.load(f)
            except Exception: pass
        elif os.path.exists(alt_json_path):
            try:
                with open(alt_json_path, 'r', encoding='utf-8') as f: doc_data_preview = json.load(f)
            except Exception: pass

        config_dir = "config/groups"
        csv_files = [f for f in os.listdir(config_dir) if f.endswith('.csv')] if os.path.exists(config_dir) else []
        default_csv_index = 0
        if csv_files and "filiere" in doc_data_preview:
            filiere_name = doc_data_preview.get("filiere", {}).get("value", "")
            if filiere_name:
                mapped_csv = map_filiere_to_csv(filiere_name)
                if mapped_csv in csv_files:
                    default_csv_index = csv_files.index(mapped_csv)
                    
        if csv_files:
            selected_csv = st.sidebar.selectbox("Base de données Étudiants (CSV) :", csv_files, index=default_csv_index)
            if default_csv_index != 0:
                st.sidebar.caption(f"CSV auto-sélectionné par l'OCR.")
        else:
            st.sidebar.warning("Aucun CSV d'étudiants trouvé dans config/groups.")
            selected_csv = None

    # -------------------
    # TABS
    # -------------------
    tab_docs, tab_orch, tab_meta, tab_abs, tab_vis, tab_json = st.tabs([
        "1. Docs Bruts", 
        "2. Orchestrateur", 
        "3. Métadonnées", 
        "4. Absences", 
        "5. Vérif. Scan",
        "6. Visionneuse JSON"
    ])

    # -- TAB 1: GESTION DES DOCUMENTS --
    with tab_docs:
        st.subheader("Uploader de nouvelles images")
        col_new1, col_new2 = st.columns([1, 2])
        with col_new1:
            new_doc_id = st.text_input("ID du Document (ex: doc_20)")
        with col_new2:
            uploaded_files = st.file_uploader("Sélectionnez les feuilles d'absences", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
            
        if st.button("Sauvegarder le Document", type="primary"):
            if new_doc_id and uploaded_files:
                doc_path = os.path.join(raw_dir, new_doc_id)
                os.makedirs(doc_path, exist_ok=True)
                for uf in uploaded_files:
                    with open(os.path.join(doc_path, uf.name), "wb") as f:
                        f.write(uf.getbuffer())
                st.success(f"Document '{new_doc_id}' créé avec {len(uploaded_files)} image(s).")
                st.rerun()
            else:
                st.warning("Veuillez fournir un ID de document et uploader au moins une image.")
                
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
                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            if st.button("Pivoter 90°", key=f"rot_{manage_doc}_{img_file}"):
                                with Image.open(img_p) as im:
                                    im = im.rotate(-90, expand=True)
                                    im.save(img_p)
                                st.rerun()
                        with col_btn2:
                            if st.button("Supprimer", key=f"del_{manage_doc}_{img_file}"):
                                os.remove(img_p)
                                st.rerun()
            else:
                st.info("Aucune image dans ce document.")
                if st.button("Supprimer le dossier vide", type="primary"):
                    os.rmdir(doc_path)
                    st.rerun()

    # If no doc selected, stop rendering
    if not selected_doc:
        st.stop()

    # -- TAB 2: ORCHESTRATEUR DE PIPELINE --
    with tab_orch:
        st.subheader("Orchestrateur & Débogage du Pipeline")
        st.write("Exécutez le pipeline entier d'un clic, ou étape par étape pour inspecter la vision par ordinateur.")
        
        if st.button("🚀 Lancer le Pipeline Complet (Mode Auto)", type="primary", use_container_width=True):
            with st.spinner("Exécution automatique de toutes les étapes..."):
                res_hdr = subprocess.run(["python", "src/ocr_header.py", selected_doc], capture_output=True, text=True)
                if res_hdr.returncode != 0: st.error(f"Échec OCR En-tête:\n{res_hdr.stderr}")
                
                std_args = ["python", "src/ocr_students.py", selected_doc]
                if selected_csv: std_args.extend(["--csv", selected_csv])
                res_std = subprocess.run(std_args, capture_output=True, text=True)
                if res_std.returncode != 0: st.error(f"Échec OCR Étudiants:\n{res_std.stderr}")
                
                if res_hdr.returncode == 0 and res_std.returncode == 0:
                    st.success("Pipeline complet terminé avec succès !")
                    st.rerun()
                    
        st.markdown("---")
        st.write("### Exécution Manuelle (Étape par Étape)")
        
        # Helper to display gallery
        def show_gallery(directory, width=200, limit=20):
            if os.path.exists(directory):
                imgs = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png'))]
                if imgs:
                    st.image([os.path.join(directory, img) for img in imgs[:limit]], width=width, caption=imgs[:limit])
                    if len(imgs) > limit: st.caption(f"... et {len(imgs)-limit} autres images masquées.")
                else: st.info("Dossier vide.")
            else: st.info("Dossier non généré.")

        # Step 1
        st.markdown("#### Étape 1 : Nettoyage & Redressement (Preprocessing)")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("Exécuter Preprocessing", key="run_prep"):
                with st.spinner("Nettoyage en cours..."):
                    res = subprocess.run(["python", "src/preprocessing.py", selected_doc], capture_output=True, text=True)
                    if res.returncode == 0: st.success("Terminé") 
                    else: st.error(res.stderr)
        with c2:
            show_gallery(os.path.join("data", "preprocessed", selected_doc), width=150)
            
        # Step 2
        st.markdown("#### Étape 2 : Découpage En-tête/Tableau (Cropper)")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("Exécuter Découpage", key="run_crop"):
                with st.spinner("Découpage en cours..."):
                    res = subprocess.run(["python", "src/cropper.py", selected_doc], capture_output=True, text=True)
                    if res.returncode == 0: st.success("Terminé") 
                    else: st.error(res.stderr)
        with c2:
            show_gallery(os.path.join("data", "cropped", selected_doc), width=250)
            
        # Step 3
        st.markdown("#### Étape 3 : Isolation des Étudiants (Row Slicer)")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("Exécuter Lignes", key="run_rows"):
                with st.spinner("Génération des lignes..."):
                    res = subprocess.run(["python", "src/row_slicer.py", selected_doc], capture_output=True, text=True)
                    if res.returncode == 0: st.success("Terminé") 
                    else: st.error(res.stderr)
        with c2:
            show_gallery(os.path.join("data", "rows", selected_doc), width=350, limit=10)

        # Step 4
        st.markdown("#### Étape 4 : Extraction du Contexte (OCR En-tête)")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("Exécuter OCR En-tête", key="run_ocr_head"):
                with st.spinner("Lecture de l'en-tête..."):
                    res = subprocess.run(["python", "src/ocr_header.py", selected_doc], capture_output=True, text=True)
                    if res.returncode == 0: st.success("Terminé") 
                    else: st.error(res.stderr)
        with c2:
            p = os.path.join("data", "output", selected_doc, "metadata.json")
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f: st.json(json.load(f), expanded=False)
            else: st.info("JSON non généré.")

        # Step 5
        st.markdown("#### Étape 5 : Extraction des Présences (OCR Étudiants)")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("Exécuter OCR Étudiants", key="run_ocr_std"):
                with st.spinner("Scan des signatures et cases..."):
                    std_args = ["python", "src/ocr_students.py", selected_doc]
                    if selected_csv: std_args.extend(["--csv", selected_csv])
                    res = subprocess.run(std_args, capture_output=True, text=True)
                    if res.returncode == 0: st.success("Terminé") 
                    else: st.error(res.stderr)
        with c2:
            p = os.path.join("data", "output", selected_doc, "absences.json")
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f: st.json(json.load(f), expanded=False)
            else: st.info("JSON non généré.")

    # -------------------
    # LOAD DOCUMENT DATA
    # -------------------
    json_path = os.path.join("data", "output", f"{selected_doc}.json")
    alt_json_path = os.path.join("data", "output", selected_doc, "metadata.json")
    absences_path = os.path.join("data", "output", selected_doc, "absences.json")

    doc_data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f: doc_data = json.load(f)
        except Exception: pass
    elif os.path.exists(alt_json_path) and os.path.exists(absences_path):
        try:
            with open(alt_json_path, 'r', encoding='utf-8') as f: doc_data = json.load(f)
            with open(absences_path, 'r', encoding='utf-8') as f: doc_data["absences"] = json.load(f)
        except Exception: pass

    # -----------------------------------
    # TAB 3: Header Metadata Review
    # -----------------------------------
    updated_meta = {}
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
                if isinstance(val_obj, dict):
                    val = val_obj.get("value", "")
                    conf = val_obj.get("confidence", 0.0)
                else:
                    val = str(val_obj)
                    conf = 0.0
                display_key = "Filière" if key == "filiere" else "Année" if key == "annee" else key.capitalize()
                with target_col:
                    new_val = st.text_input(display_key, value=val, key=f"meta_{key}")
                    updated_meta[key] = new_val
                    if conf < 0.5: st.warning(f"Confiance : {conf*100:.1f}% - Vérifiez.")
                    else: st.caption(f"Confiance : {conf*100:.1f}%")

    # -----------------------------------
    # TAB 4: Attendance Review
    # -----------------------------------
    with tab_abs:
        if not doc_data or "absences" not in doc_data:
            st.info("Lancez d'abord l'OCR Étudiants dans l'Orchestrateur.")
        else:
            st.subheader("Liste de Présence des Étudiants")
            absences = doc_data.get("absences", [])
            if not absences:
                st.info("Aucune absence d'étudiant trouvée.")
            else:
                df_data = []
                for r in absences:
                    row = {
                        "N° Apo": r.get('n_apo', ''),
                        "Nom": r.get('nom', ''),
                        "Prénom": r.get('prenom', ''),
                        "Confiance Assoc": r.get('match_confidence', 0.0),
                    }
                    sessions = r.get('sessions', {})
                    low_conf_seances = []
                    for i in range(1, 11):
                        s_key = f"seance{i}"
                        s_data = sessions.get(s_key, {})
                        is_present = s_data.get("is_present", False)
                        conf = s_data.get("confidence", 0.0)
                        row[f"Séance {i}"] = bool(is_present)
                        if conf > 0 and conf < 40.0:
                            low_conf_seances.append(str(i))
                    row["Nécessite Révision"] = "Séances : " + ", ".join(low_conf_seances) if low_conf_seances else ""
                    df_data.append(row)
                    
                df = pd.DataFrame(df_data)
                column_config = {
                    "N° Apo": st.column_config.TextColumn("N° Apo", disabled=True),
                    "Nom": st.column_config.TextColumn("Nom", disabled=True),
                    "Prénom": st.column_config.TextColumn("Prénom", disabled=True),
                    "Confiance Assoc": st.column_config.NumberColumn("Confiance (%)", disabled=True, format="%.1f"),
                    "Nécessite Révision": st.column_config.TextColumn("Nécessite Révision", disabled=True),
                }
                for col in [f"Séance {i}" for i in range(1, 11)]:
                    column_config[col] = st.column_config.CheckboxColumn(col)
                    
                edited_df = st.data_editor(
                    df, column_config=column_config, hide_index=True, use_container_width=True, num_rows="fixed"
                )
                
                st.markdown("---")
                if st.button("Enregistrer les données vérifiées", type="primary"):
                    student_csv = doc_data.get("student_list_csv", selected_csv)
                    final_path = save_verified_data(selected_doc, updated_meta, edited_df, student_csv)
                    st.success(f"Données finales enregistrées dans : {final_path}")

    # -----------------------------------
    # TAB 5: Physical Sheet Verification
    # -----------------------------------
    with tab_vis:
        st.subheader("Visionneuse Interactive de Cellules")
        debug_dir = os.path.join("debug", selected_doc)
        absences = doc_data.get("absences", []) if doc_data else []
        if os.path.exists(debug_dir) and absences:
            col_s, col_seance = st.columns(2)
            student_options = [f"Ligne {r.get('row_index')} - {r.get('nom')} {r.get('prenom')}" for r in absences if r.get('row_index')]
            if student_options:
                selected_student = col_s.selectbox("Sélectionnez l'étudiant :", student_options)
                selected_seance = col_seance.selectbox("Sélectionnez la séance :", range(1, 11))
                row_idx = selected_student.split(" - ")[0].replace("Ligne ", "")
                img_path = os.path.join(debug_dir, f"row{row_idx}_seance{selected_seance}.jpg")
                if os.path.exists(img_path):
                    st.image(Image.open(img_path), width=300, caption=f"Ligne {row_idx}, Séance {selected_seance}")
                else:
                    st.warning("Image introuvable.")
            else: st.info("Aucune ligne physique.")
        else: st.info("Images non disponibles. Exécutez l'Orchestrateur avec le débogage actif.")

    # -----------------------------------
    # TAB 6: JSON Viewer
    # -----------------------------------
    with tab_json:
        st.subheader("Visionneuse JSON (Base de Données)")
        json_choice = st.radio("Sélectionnez le fichier à inspecter :", [
            "1. Sortie OCR En-tête (metadata.json)", 
            "2. Sortie OCR Étudiants (absences.json)", 
            "3. Export Final Compilé (verified_output/)"
        ])
        
        if "metadata.json" in json_choice:
            p = os.path.join("data", "output", selected_doc, "metadata.json")
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f: st.json(json.load(f))
            else: st.info("Fichier non généré.")
            
        elif "absences.json" in json_choice:
            p = os.path.join("data", "output", selected_doc, "absences.json")
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f: st.json(json.load(f))
            else: st.info("Fichier non généré.")
            
        elif "Export Final" in json_choice:
            if not updated_meta:
                st.info("Veuillez réviser les métadonnées dans l'onglet correspondant.")
            else:
                filiere = updated_meta.get("filiere", "Inconnue").strip()
                annee = updated_meta.get("annee", "Inconnue").strip()
                safe_filiere = re.sub(r'[^a-zA-Z0-9_\-]', '_', filiere)
                safe_annee = re.sub(r'[^a-zA-Z0-9_\-]', '_', annee)
                p = os.path.join("verified_output", f"Absences_{safe_filiere}_{safe_annee}.json")
                if os.path.exists(p):
                    with open(p, 'r', encoding='utf-8') as f: st.json(json.load(f))
                else: st.info(f"Fichier final '{p}' non trouvé. Avez-vous cliqué sur 'Enregistrer' ?")

if __name__ == "__main__":
    main()
