import streamlit as st
import pandas as pd
import json
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from pipeline import run

st.set_page_config(
    page_title="Gestion des Absences — PoC",
    page_icon="",
    layout="wide"
)

st.title("Automatisation de la Saisie des Absences")
st.caption("Pipeline de reconnaissance de documents — Proof of Concept")

# ── Sidebar: tunable parameters ──────────────────────────────────────────────
st.sidebar.header("Paramètres du Pipeline")
header_ratio = st.sidebar.slider(
    "Ratio en-tête", min_value=0.10, max_value=0.40, value=0.22, step=0.01
)
col_ratio = st.sidebar.slider(
    "Position colonne signature", min_value=0.60, max_value=0.95, value=0.85, step=0.01
)
density_threshold = st.sidebar.slider(
    "Seuil densité pixels (présence/absence)",
    min_value=0.005, max_value=0.10, value=0.02, step=0.005
)

# ── File upload ───────────────────────────────────────────────────────────────
st.subheader("1. Uploader la feuille d'absence")
uploaded_image = st.file_uploader("Feuille scannée (JPG, PNG)", type=["jpg", "jpeg", "png"])
uploaded_db = st.file_uploader("Liste étudiants (CSV avec colonne 'name')", type=["csv"])

# ── Run pipeline ──────────────────────────────────────────────────────────────
if uploaded_image and uploaded_db:
    st.subheader("2. Image uploadée")
    st.image(uploaded_image, caption="Feuille scannée originale", width=600)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_tmp:
        img_tmp.write(uploaded_image.read())
        img_path = img_tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as db_tmp:
        db_tmp.write(uploaded_db.read())
        db_path = db_tmp.name

    with st.spinner("Traitement en cours..."):
        try:
            result = run(
                img_path, db_path,
                header_ratio=header_ratio,
                col_ratio=col_ratio,
                density_threshold=density_threshold
            )

            # Header fields display
            st.subheader("3. Informations extraites de l'en-tête")
            header = result["header_fields"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Module", header.get("module") or "Non détecté")
            col1.metric("Enseignant", header.get("teacher") or "Non détecté")
            col2.metric("Date", header.get("date") or "Non détecté")
            col2.metric("Horaire", header.get("time") or "Non détecté")
            col3.metric("Type de séance", header.get("session_type") or "Non détecté")

            # Absences display
            st.subheader("4. Étudiants absents détectés")
            absent = result["absent_students"]
            if absent:
                df_absent = pd.DataFrame({
                    "N° ligne": result["absent_indices"],
                    "Étudiant absent": absent
                })
                st.dataframe(df_absent, use_container_width=True)
            else:
                st.success("Aucune absence détectée.")

            # Export
            st.subheader("5. Exporter les résultats")
            c1, c2 = st.columns(2)

            csv_data = pd.DataFrame({
                "module": [header.get("module")],
                "teacher": [header.get("teacher")],
                "date": [header.get("date")],
                "time": [header.get("time")],
                "session_type": [header.get("session_type")],
                "absent_students": [", ".join(absent)]
            }).to_csv(index=False).encode("utf-8")

            c1.download_button("Télécharger CSV", csv_data, "absences.csv", "text/csv")

            json_data = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
            c2.download_button("Télécharger JSON", json_data, "absences.json", "application/json")

        except Exception as e:
            st.error(f"Erreur lors du traitement: {str(e)}")

        finally:
            os.unlink(img_path)
            os.unlink(db_path)

else:
    st.info("Veuillez uploader une feuille d'absence et le fichier étudiants pour continuer.")