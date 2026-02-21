"""Paths and hyperparameters for Malaria Detection."""
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "cell_images"
MODELS_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results"

# Classes
CLASS_NAMES = ["Parasitized", "Uninfected"]
NUM_CLASSES = len(CLASS_NAMES)

# Image
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 42
