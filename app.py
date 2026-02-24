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
    initial_sidebar_state="expanded",
)

# ---------- ADVANCED STYLES ----------

st.markdown("""
<style>
    /* Typography & layout */
    .main-header {
        font-size: 2.25rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label {
        color: #e2e8f0 !important;
    }
    /* Result cards */
    .result-infected {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 50%, #b91c1c 100%);
        padding: 1.75rem;
        border-radius: 16px;
        color: #fef2f2;
        font-size: 1.35rem;
        font-weight: 700;
        box-shadow: 0 20px 50px rgba(185, 28, 28, 0.35);
        border: 1px solid rgba(254, 226, 226, 0.15);
    }
    .result-uninfected {
        background: linear-gradient(135deg, #14532d 0%, #166534 50%, #15803d 100%);
        padding: 1.75rem;
        border-radius: 16px;
        color: #dcfce7;
        font-size: 1.35rem;
        font-weight: 700;
        box-shadow: 0 20px 50px rgba(22, 163, 74, 0.3);
        border: 1px solid rgba(220, 252, 231, 0.2);
    }
    .result-confidence {
        font-size: 1.75rem;
        font-weight: 800;
        display: block;
        margin-top: 0.5rem;
        letter-spacing: 0.02em;
    }
    /* Prediction summary card - full width, no truncation */
    .summary-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
    }
    .summary-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }
    .summary-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f1f5f9;
        word-break: break-word;
    }
    .summary-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        align-items: start;
    }
    .summary-item {
        min-width: 0;
    }
    /* File uploader area */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
        overflow: hidden;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)
def is_likely_blood_cell_image(img: np.ndarray) -> bool:
    """
    Simple heuristic check: image has minimum size and color profile
    typical of microscopic blood smear (red/purple tones, not grayscale).
    """
    if img is None or img.size == 0:
        return False

    h, w = img.shape[:2]
    # Reject very small or thumbnail-like images
    if min(h, w) < 40:
        return False

    # Reject extreme aspect ratios (e.g. documents, banners)
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > 6:
        return False

    # Blood smear images (e.g. Giemsa) usually have red/purple tones
    # and are not pure grayscale
    b, g, r = cv2.split(img)
    r_mean, g_mean, b_mean = float(r.mean()), float(g.mean()), float(b.mean())

    # Grayscale check: R≈G≈B => likely not a stained smear
    gray_score = abs(r_mean - g_mean) + abs(g_mean - b_mean) + abs(b_mean - r_mean)
    if gray_score < 25:
        return False

    # Some presence of red (blood/stain)
    if r_mean < 30 and g_mean < 30 and b_mean < 30:
        return False

    return True


def get_available_models():
    """List saved models (best.keras or final.keras), excluding 'custom'."""
    models = []
    if not MODELS_DIR.exists():
        return models
    for d in MODELS_DIR.iterdir():
        if not d.is_dir():
            continue
        if d.name.lower() == "custom":
            continue
        for name in ("best.keras", "final.keras"):
            p = d / name
            if p.exists():
                models.append((d.name, str(p)))
                break
    return models


def run_app():
    st.markdown('<p class="main-header">🩸 Malaria Detection System</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Automated parasite detection & image classification for healthcare support</p>',
        unsafe_allow_html=True,
    )

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
            "No trained model found. Add trained model files (e.g. best.keras or final.keras) "
            "inside the models directory."
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

                    # Prediction summary - custom card so every word is visible (no truncation)
                    st.markdown("##### Prediction summary")
                    st.markdown(
                        f'''
                        <div class="summary-card">
                            <div class="summary-row">
                                <div class="summary-item">
                                    <div class="summary-title">Predicted class</div>
                                    <div class="summary-value">{name}</div>
                                </div>
                                <div class="summary-item">
                                    <div class="summary-title">Confidence</div>
                                    <div class="summary-value">{conf:.1%}</div>
                                </div>
                                <div class="summary-item">
                                    <div class="summary-title">Model</div>
                                    <div class="summary-value">{model_choice}</div>
                                </div>
                            </div>
                        </div>
                        ''',
                        unsafe_allow_html=True,
                    )

            else:
                st.error("Could not decode image. Please upload a valid PNG/JPG image.")
        elif uploaded is not None and not model_path:
            st.warning("No trained model is available. Please add a trained model to the models directory.")
        else:
            st.info("Upload a cell image to start diagnosis.")

    # ---------- TAB 2: ACCURACY ANALYSIS ----------
    with tabs[1]:
        st.subheader("Model accuracy analysis")
        st.caption(
            "Latest saved evaluation results. Generated offline and stored in the results folder."
        )

        report_file = RESULTS_DIR / "classification_report.txt"
        cm_path = RESULTS_DIR / "confusion_matrix.png"
        roc_path = RESULTS_DIR / "roc_curve.png"

        if not (report_file.exists() or cm_path.exists() or roc_path.exists()):
            st.info(
                "No evaluation artifacts were found in the results folder. "
                "Run the evaluation script locally and commit the generated report and plots."
            )
        else:
            col_cm, col_roc = st.columns(2)
            if cm_path.exists():
                with col_cm:
                    st.markdown("##### Confusion Matrix")
                    st.image(str(cm_path), use_container_width=True)
            if roc_path.exists():
                with col_roc:
                    st.markdown("##### ROC Curve")
                    st.image(str(roc_path), use_container_width=True)

            if report_file.exists():
                with st.expander("Detailed classification report", expanded=True):
                    with open(report_file) as f:
                        st.code(f.read(), language="text")


if __name__ == "__main__":
    run_app()

