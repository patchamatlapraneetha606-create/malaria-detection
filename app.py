"""
Malaria Detection - Healthcare Support System (Streamlit UI).
Automated parasite detection, image classification, and model accuracy analysis.
"""
import streamlit as st
import numpy as np
from pathlib import Path
import cv2
from tensorflow import keras

from config import MODELS_DIR, RESULTS_DIR, RAW_DATA_DIR, CLASS_NAMES, IMG_SIZE
from data_loader import load_image_opencv, preprocess_opencv, get_keras_image_dataset
from models import get_model
from predict import load_trained_model, preprocess_image, predict_single

st.set_page_config(
    page_title="Malaria Detection System",
    page_icon="🩸",
    layout="wide",
)

# Custom styling for healthcare look
st.markdown("""
<style>
    .main-header { font-size: 2rem; color: #1e3a5f; margin-bottom: 0.5rem; }
    .sub-header { color: #4a6fa5; margin-bottom: 1.5rem; }
    .result-infected { background: #ffebee; padding: 1rem; border-radius: 8px; border-left: 4px solid #c62828; }
    .result-uninfected { background: #e8f5e9; padding: 1rem; border-radius: 8px; border-left: 4px solid #2e7d32; }
    .metric-box { background: #f5f5f5; padding: 1rem; border-radius: 8px; text-align: center; }
    div[data-testid="stSidebar"] { background: #fafafa; }
</style>
""", unsafe_allow_html=True)


def get_available_models():
    """List saved models (best.keras or final.keras)."""
    models = []
    if not MODELS_DIR.exists():
        return models
    for d in MODELS_DIR.iterdir():
        if d.is_dir():
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
        sidebar.info("No trained model found. Train a model in the **Train** tab first.")

    tabs = st.tabs(["🔬 Diagnose", "📊 Accuracy Analysis", "🏋️ Train Model"])

    # Tab 1: Diagnose (upload image, get prediction)
    with tabs[0]:
        st.subheader("Automated diagnosis")
        uploaded = st.file_uploader("Upload a blood smear cell image", type=["png", "jpg", "jpeg"])
        if uploaded is not None and model_path:
            img_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                col1, col2 = st.columns(2)
                with col1:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, caption="Uploaded image", use_container_width=True)
                with col2:
                    model = load_trained_model(Path(model_path))
                    idx, name, conf, probs = predict_single(model, img)
                    if name == "Parasitized":
                        st.markdown(
                            f'<div class="result-infected"><strong>Result: Parasitized (infected)</strong><br>'
                            f'Confidence: {conf:.1%}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="result-uninfected"><strong>Result: Uninfected</strong><br>'
                            f'Confidence: {conf:.1%}</div>',
                            unsafe_allow_html=True,
                        )
                    st.write("Probabilities:")
                    for c, p in zip(CLASS_NAMES, probs):
                        st.progress(float(p), text=f"{c}: {p:.1%}")
            else:
                st.error("Could not decode image.")
        elif uploaded is not None and not model_path:
            st.warning("Train a model first, then run diagnosis.")

    # Tab 2: Accuracy analysis (show report and plots if available)
    with tabs[1]:
        st.subheader("Model accuracy analysis")
        if model_path:
            if st.button("Check accuracy", type="primary", help="Run evaluation on the selected model"):
                try:
                    from evaluate import run_analysis
                    with st.spinner("Evaluating model accuracy..."):
                        run_analysis(Path(model_path), RAW_DATA_DIR, RESULTS_DIR)
                    st.success("Evaluation complete.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
            st.divider()
        else:
            st.warning("Select a trained model in the sidebar first, then check accuracy.")
        report_file = RESULTS_DIR / "classification_report.txt"
        if report_file.exists():
            with open(report_file) as f:
                st.text(f.read())
            col1, col2 = st.columns(2)
            cm_path = RESULTS_DIR / "confusion_matrix.png"
            roc_path = RESULTS_DIR / "roc_curve.png"
            if cm_path.exists():
                with col1:
                    st.image(str(cm_path), caption="Confusion Matrix", use_container_width=True)
            if roc_path.exists():
                with col2:
                    st.image(str(roc_path), caption="ROC Curve", use_container_width=True)
        else:
            st.info(
                "Click **Check accuracy** above to run evaluation and generate the classification report, "
                "confusion matrix, and ROC curve."
            )

    # Tab 3: Train model
    with tabs[2]:
        st.subheader("Train model")
        data_dir = RAW_DATA_DIR.resolve()
        if not data_dir.exists():
            st.warning(
                f"Data directory not found: `{data_dir}`. Create it with subfolders "
                "**Parasitized** and **Uninfected** containing cell images."
            )
            if st.button("Download sample data (1000 per class; first time downloads ~337 MB)"):
                import subprocess
                import sys
                cwd = Path(__file__).resolve().parent
                with st.spinner("Downloading malaria sample images (may take a few minutes)..."):
                    out = subprocess.run(
                        [sys.executable, str(cwd / "download_data.py"), "--max-per-class", "1000"],
                        capture_output=True, text=True, cwd=str(cwd),
                        timeout=900,
                    )
                if out.returncode == 0:
                    st.success("Sample data downloaded. Refresh the page to start training.")
                    st.rerun()
                else:
                    st.error("Download failed: " + (out.stderr or out.stdout or "")[:500])
        else:
            train_ds, val_ds, _ = get_keras_image_dataset(data_dir)
            if train_ds is None:
                st.warning("No images found in Parasitized/ and Uninfected/. Download sample data first.")
                if st.button("Download sample data (1000 per class; first time downloads ~337 MB)"):
                    import subprocess
                    import sys
                    cwd = Path(__file__).resolve().parent
                    with st.spinner("Downloading malaria sample images (may take a few minutes)..."):
                        out = subprocess.run(
                            [sys.executable, str(cwd / "download_data.py"), "--max-per-class", "1000"],
                            capture_output=True, text=True, cwd=str(cwd),
                            timeout=900,
                        )
                    if out.returncode == 0:
                        st.success("Sample data downloaded. You can start training now.")
                        st.rerun()
                    else:
                        st.error("Download failed: " + (out.stderr or out.stdout or "")[:400])
            else:
                st.info("Training data loaded. Choose architecture and epochs, then click Start training.")
                model_name = st.selectbox(
                    "Architecture",
                    ["custom", "mobilenetv2", "efficientnet"],
                    index=0,
                    help="Use 'custom' if mobilenetv2/efficientnet fail (e.g. SSL errors).",
                )
                epochs = st.slider("Epochs", 2, 30, 10)
                if st.button("Start training"):
                    import subprocess
                    import sys
                    cwd = Path(__file__).resolve().parent
                    cmd = [
                        sys.executable, str(cwd / "train.py"),
                        "--model", model_name,
                        "--epochs", str(epochs),
                    ]
                    with st.spinner("Training in progress (this may take several minutes)..."):
                        out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))
                    if out.returncode == 0:
                        st.success("Training complete. You can use **Diagnose** and **Accuracy Analysis**.")
                    else:
                        st.error("Training failed. Check terminal: " + (out.stderr or out.stdout or "")[:500])
                    if st.button("Refresh page"):
                        st.rerun()


if __name__ == "__main__":
    run_app()
