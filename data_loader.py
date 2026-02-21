"""Dataset loading and OpenCV preprocessing."""
import numpy as np
import cv2
from pathlib import Path
from tensorflow import keras

from config import RAW_DATA_DIR, CLASS_NAMES, IMG_SIZE, BATCH_SIZE, VAL_SPLIT, SEED


def load_image_opencv(path: Path) -> np.ndarray:
    """Load image with OpenCV (BGR)."""
    img = cv2.imread(str(path))
    return img


def preprocess_opencv(img: np.ndarray, target_size=IMG_SIZE) -> np.ndarray:
    """Resize and normalize for model input. Returns RGB float32 [0,1]."""
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def get_keras_image_dataset(
    data_dir: Path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    val_split=VAL_SPLIT,
    seed=SEED,
):
    """
    Build train/val datasets from data_dir/Parasitized and data_dir/Uninfected.
    Returns (train_ds, val_ds, class_names) or (None, None, CLASS_NAMES) if no images.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return None, None, CLASS_NAMES

    try:
        train_ds = keras.utils.image_dataset_from_directory(
            data_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=CLASS_NAMES,
            color_mode="rgb",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            validation_split=val_split,
            subset="training",
        )
        val_ds = keras.utils.image_dataset_from_directory(
            data_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=CLASS_NAMES,
            color_mode="rgb",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            validation_split=val_split,
            subset="validation",
        )
        norm = keras.layers.Rescaling(1.0 / 255.0)
        train_ds = train_ds.map(lambda x, y: (norm(x), y))
        val_ds = val_ds.map(lambda x, y: (norm(x), y))
        return train_ds, val_ds, CLASS_NAMES
    except (ValueError, OSError, TypeError):
        return None, None, CLASS_NAMES
