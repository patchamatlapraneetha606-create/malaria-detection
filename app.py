"""
Malaria Detection - Healthcare Support System (Streamlit UI).
Automated parasite detection, image classification, and model accuracy analysis.
"""
import streamlit as st
import numpy as np
from pathlib import Path
import cv2

from config import MODELS_DIR, RESULTS_DIR, CLASS_NAMES, IMG_SIZE
from predict import load_trained_model, predict_single

st.set_page_config(
    page_title="Malaria Detection System",
    page_icon="🩸",
    layout="wide",
)

# ---------- STYLES ----------

st.markdown("""
<style>
    .main-header { font-size: 2rem; color: #f5f5f5; margin-bottom: 0.25rem; font-weight: 700; }
    .sub-header { color: #b0bec5; margin-bottom: 1.5rem; }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #020617 100%);
        color: #e5e7eb;
    }
    div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] label {
        color: #e5e7eb !important;
    }
    .result-infected {
        background: radial-gradient(circle at top left, #ff8a80 0, #b71c1c 55%);
        padding: 1.5rem;
        border-radius: 14px;
        color: #fff;
        font-size: 1.3rem;
        font-weight: 700;
        box-shadow: 0 18px 40px rgba(239, 68, 68, 0.45);
    }
    .result-uninfected {
        background: radial-gradient(circle at top left, #a5d6a7 0, #1b5e20 55%);
        padding: 1.5rem;
        border-radius: 14px;
        color: #e8f5e9;
        font-size: 1.3rem;
        font-weight: 700;
        box-shadow: 0 18px 40px rgba(34, 197, 94, 0.4);
    }
    .result-confidence {
        font-size: 1.6rem;
        font-weight: 800;
        display: block;
        margin-top: 0.5rem;
    }
    .metric-chip {
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.5);
        display: inline-block;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------- MODEL DISCOVERY ----------

def get_available_models():
    """List saved models (best.keras or final.keras), excluding 'custom'."""
    models = []
    if not MODELS_DIR.exists():
        return models
    for d in MODELS_DIR.iterdir():
        if not d.is_dir():
            continue

        # Hide 'custom' from the UI dropdown
        if d.name.lower() == "custom":
            continue

        for name in ("best.keras", "final.keras"):
            p = d / name
            if p.exists():
                models.append((d.name, str(p)))
                break
    return models


# ---------- APP ----------

def run_app():
    st.markdown('<p class="main-header">🩸 Malaria Detection System</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Automated parasite detection & image classification for healthcare support</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    sidebar = st.sidebar
    sidebar.header("Settings")
    available = get_available_models()
    if available:
        model_choice = sidebar.selectbox(
            "Model",
            options=[m[0] for m in available],
            index=0,
        )
        model_path = next(p for n, p in available if n == model_choice)
    else:
        model_path = None
        sidebar.info(
            "No trained model found.\n\n"
            "Add trained model files (e.g. best.keras or final.keras) inside the models directory."
        )

    tabs = st.tabs(["🔬 Diagnose", "📊 Accuracy Analysis"])

    # ---------- TAB 1: DIAGNOSE ----------
    with tabs[0]:
        st.subheader("Automated diagnosis")
        st.caption("Upload a single-cell blood smear image to obtain an instant prediction.")

        uploaded = st.file_uploader(
            "Upload a blood smear cell image",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded is not None and model_path:
            img_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

            if img is not None:
                col_img, col_result = st.columns([1.1, 1])

                with col_img:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, caption="Uploaded image", use_container_width=True)

                with col_result:
                    model = load_trained_model(Path(model_path))
                    idx, name, conf, probs = predict_single(model, img)

                    if name == "Parasitized":
                        st.markdown(
                            f'<div class="result-infected">Parasitized (infected)'
                            f'<span class="result-confidence">{conf:.1%} confidence</span></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="result-uninfected">Uninfected'
                            f'<span class="result-confidence">{conf:.1%} confidence</span></div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("##### Class probabilities")
                    for c, p in zip(CLASS_NAMES, probs):
                        st.progress(float(p), text=f"{c}: {p:.1%}")

                    st.markdown("##### Prediction summary")
                    cols = st.columns(3)
                    cols[0].metric("Predicted class", name)
                    cols[1].metric("Confidence", f"{conf:.1%}")
                    cols[2].metric("Model", model_choice)

            else:
                st.error("Could not decode image. Please upload a valid PNG/JPG image.")
        elif uploaded is not None and not model_path:
            st.warning("No trained model is available. Please add a trained model to the models directory.")
        else:
            st.info("Upload a cell image on the left to start diagnosis.")

    # ---------- TAB 2: ACCURACY ANALYSIS (read‑only, single view) ----------
    with tabs[1]:
        st.subheader("Model accuracy analysis")
        st.caption(
            "Below are the latest saved evaluation results for the selected model. "
            "These are generated offline and stored in the results folder."
        )

        report_file = RESULTS_DIR / "classification_report.txt"
        cm_path = RESULTS_DIR / "confusion_matrix.png"
        roc_path = RESULTS_DIR / "roc_curve.png"

        if not (report_file.exists() or cm_path.exists() or roc_path.exists()):
            st.info(
                "No evaluation artifacts were found in the results folder.\n\n"
                "To populate this section, run the evaluation script locally and "
                "commit the generated classification report and plots."
            )
        else:
            # Top row: confusion matrix + ROC
            col_cm, col_roc = st.columns(2)
            if cm_path.exists():
                with col_cm:
                    st.markdown("##### Confusion Matrix")
                    st.image(str(cm_path), use_container_width=True)
            if roc_path.exists():
                with col_roc:
                    st.markdown("##### ROC Curve")
                    st.image(str(roc_path), use_container_width=True)

            # Detailed text report in an expander (single occurrence)
            if report_file.exists():
                with st.expander("Detailed classification report", expanded=True):
                    with open(report_file) as f:
                        st.code(f.read(), language="text")


if __name__ == "__main__":
    run_app()
