"""
Malaria Detection - Healthcare Support System (Streamlit UI).
Automated parasite detection, image classification, and model accuracy analysis.
"""
import streamlit as st
import numpy as np
from pathlib import Path
import cv2
from tensorflow import keras
import tensorflow as tf

from config import MODELS_DIR, RAW_DATA_DIR, CLASS_NAMES, IMG_SIZE
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
    .result-infected { background: #ffebee; padding: 1.25rem; border-radius: 8px; border-left: 5px solid #c62828; color: #0d0d0d; font-size: 1.35rem; font-weight: 700; }
    .result-uninfected { background: #e8f5e9; padding: 1.25rem; border-radius: 8px; border-left: 5px solid #2e7d32; color: #0d0d0d; font-size: 1.35rem; font-weight: 700; }
    .result-confidence { font-size: 1.5rem; font-weight: 800; }
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


def evaluate_model_on_validation(model_path: Path):
    """
    Evaluate the selected model on a validation split created from RAW_DATA_DIR.

    RAW_DATA_DIR should contain subfolders for each class
    (e.g. Parasitized, Uninfected) with images.
    """
    if not RAW_DATA_DIR.exists():
        st.error(f"Data directory not found: `{RAW_DATA_DIR}`.")
        return

    try:
        val_ds = tf.keras.utils.image_dataset_from_directory(
            RAW_DATA_DIR,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=32,
        )
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

        model = load_trained_model(model_path)
        if model is None:
            st.error("Could not load the selected model.")
            return

        loss, acc = model.evaluate(val_ds, verbose=0)
    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        return

    st.success(f"Validation accuracy: {acc:.2%}")
    st.write(f"Validation loss: {loss:.4f}")


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
            "No trained model found. Please add trained model files "
            "(e.g. best.keras or final.keras) into the models directory."
        )

    # NOTE: Train Model tab REMOVED
    tabs = st.tabs(["🔬 Diagnose", "📊 Accuracy Analysis"])

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
                            f'<div class="result-infected">Result: Parasitized (infected)<br>'
                            f'<span class="result-confidence">Confidence: {conf:.1%}</span></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="result-uninfected">Result: Uninfected<br>'
                            f'<span class="result-confidence">Confidence: {conf:.1%}</span></div>',
                            unsafe_allow_html=True,
                        )
                    st.write("Probabilities:")
                    for c, p in zip(CLASS_NAMES, probs):
                        st.progress(float(p), text=f"{c}: {p:.1%}")
            else:
                st.error("Could not decode image.")
        elif uploaded is not None and not model_path:
            st.warning(
                "No trained model is available. Please add a trained model to the models directory."
            )

    # Tab 2: Accuracy analysis (simple evaluation on validation split)
    with tabs[1]:
        st.subheader("Model accuracy analysis")
        if model_path:
            st.write(
                "This evaluates the selected model on a validation split "
                "created from the raw data folder."
            )
            if st.button(
                "Check accuracy",
                type="primary",
                help="Run evaluation on the selected model",
            ):
                with st.spinner("Evaluating model accuracy..."):
                    evaluate_model_on_validation(Path(model_path))
        else:
            st.warning("Select a trained model in the sidebar first, then check accuracy.")


if __name__ == "__main__":
    run_app()
