"""Single and batch prediction (CLI and programmatic)."""
import numpy as np
import cv2
from pathlib import Path
from tensorflow import keras

from config import CLASS_NAMES, IMG_SIZE


def load_trained_model(model_path: Path) -> keras.Model:
    """Load a saved Keras model."""
    return keras.models.load_model(str(model_path))


def preprocess_image(img: np.ndarray, target_size=IMG_SIZE) -> np.ndarray:
    """Resize and normalize for model. img: BGR or RGB. Returns batch of 1."""
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def predict_single(model: keras.Model, img: np.ndarray):
    """
    Predict class for one image. img: BGR OpenCV image.
    Returns (class_index, class_name, confidence, probabilities_array).
    """
    x = preprocess_image(img)
    if x is None:
        return -1, "Unknown", 0.0, np.array([0.0, 0.0])
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return idx, CLASS_NAMES[idx], float(probs[idx]), probs
