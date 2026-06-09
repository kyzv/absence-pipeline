import streamlit as st
import os
import json
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Absence Pipeline", layout="wide")

def main():
    st.title("🎓 Système Intelligent de Gestion des Absences")
    st.markdown("---")

    # Sidebar: Setup
    st.sidebar.header("📁 Configuration")
    
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir):
        st.error(f"Dossier {raw_dir} introuvable.")
        return
        
    doc_ids = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    selected_doc = st.sidebar.selectbox("Sélectionnez le document à traiter:", doc_ids)
    
    config_dir = "data/config/groups"
    csv_files = [f for f in os.listdir(config_dir) if f.endswith('.csv')] if os.path.exists(config_dir) else []
    
    if f"{selected_doc}.csv" in csv_files:
        st.sidebar.success(f"Liste d'étudiants trouvée : {selected_doc}.csv")
    else:
        st.sidebar.warning(f"Aucune liste d'étudiants trouvée. Veuillez exécuter 'extract_students.py'.")

    # Run Pipeline Button
    if st.sidebar.button("🚀 Lancer l'Analyse"):
        with st.spinner("Analyse en cours... (Cela peut prendre quelques minutes)"):
            try:
                from src.pipeline import run_pipeline
                meta, tbl = run_pipeline(selected_doc)
                st.session_state['meta'] = meta
                st.session_state['tbl'] = tbl
                st.success("Analyse terminée avec succès !")
            except Exception as e:
                import traceback
                st.error(f"Erreur lors de l'exécution : {str(e)}\n{traceback.format_exc()}")
                return

    # Load from Output if available and not in session state
    out_dir = os.path.join("data", "output", selected_doc)
    if 'meta' not in st.session_state and os.path.exists(os.path.join(out_dir, "metadata.json")):
        with open(os.path.join(out_dir, "metadata.json"), 'r', encoding='utf-8') as f:
            st.session_state['meta'] = json.load(f)
        with open(os.path.join(out_dir, "absences.json"), 'r', encoding='utf-8') as f:
            st.session_state['tbl'] = json.load(f)

    if 'meta' in st.session_state:
        meta = st.session_state['meta']
        tbl = st.session_state['tbl']
        
        # Display Metadata
        st.subheader("📋 Métadonnées Extraites")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Filière", meta.get("filiere", "-"))
        col2.metric("Module", meta.get("module", "-"))
        col3.metric("Enseignant", meta.get("enseignant", "-"))
        col4.metric("Année", meta.get("annee", "-"))
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Date", meta.get("date", "-"))
        col2.metric("Heure Début", meta.get("heure_debut", "-"))
        col3.metric("Heure Fin", meta.get("heure_fin", "-"))
        col4.metric("Type", meta.get("type", "-").upper())
        
        st.markdown("---")
        
        # Determine available sessions
        available_sessions = set()
        for r in tbl:
            for s in r['sessions']:
                available_sessions.add(s['seance'])
        
        available_sessions = sorted(list(available_sessions), key=lambda x: int(x) if x.isdigit() else x)
        
        if not available_sessions:
            st.info("Aucune séance n'a été détectée (soit la feuille est vide, soit l'OCR a échoué).")
            return
            
        st.subheader("👥 Liste des Présences / Absences")
        selected_sessions = st.multiselect("Séances à afficher :", available_sessions, default=available_sessions)
        
        # Build DataFrame
        df_data = []
        for r in tbl:
            row = {"N° Apo": r['n_apo'], "Nom": r['nom']}
            for s in r['sessions']:
                if s['seance'] in selected_sessions:
                    row[f"Séance {s['seance']}"] = "✅ Présent" if s['status'] == "Present" else "❌ Absent"
            df_data.append(row)
            
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        # Debug Images
        st.markdown("---")
        st.subheader("🔍 Vérification (Vision par Ordinateur)")
        debug_dir = os.path.join("data", "debug")
        
        # The debug images are saved as "table_debug.jpg" inside data/debug/doc_X_page_Y/
        debug_pages = [d for d in os.listdir(debug_dir) if d.startswith(selected_doc)] if os.path.exists(debug_dir) else []
        
        if debug_pages:
            for d_page in sorted(debug_pages):
                img_path = os.path.join(debug_dir, d_page, "table_debug.jpg")
                if os.path.exists(img_path):
                    st.write(f"**{d_page}**")
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)

if __name__ == "__main__":
    main()
