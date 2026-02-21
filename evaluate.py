"""Accuracy analysis: classification report, confusion matrix, ROC curve."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
import seaborn as sns
from tensorflow import keras

from config import RAW_DATA_DIR, RESULTS_DIR, CLASS_NAMES, IMG_SIZE, BATCH_SIZE, VAL_SPLIT, SEED
from data_loader import get_keras_image_dataset
from predict import load_trained_model, preprocess_image


def run_analysis(model_path: Path, data_dir: Path, results_dir: Path):
    """
    Load model, run on validation data, write classification_report.txt,
    confusion_matrix.png, roc_curve.png to results_dir.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model = load_trained_model(model_path)
    _, val_ds, _ = get_keras_image_dataset(
        data_dir, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, seed=SEED
    )
    if val_ds is None:
        raise FileNotFoundError(f"No validation data in {data_dir}")

    y_true = []
    y_prob = []
    for x, y in val_ds:
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_prob.append(model.predict(x, verbose=0))
    y_true = np.array(y_true)
    y_prob = np.concatenate(y_prob, axis=0)
    y_pred = np.argmax(y_prob, axis=1)

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4
    )
    report_path = results_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Wrote {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    cm_path = results_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=100)
    plt.close()
    print(f"Wrote {cm_path}")

    # ROC (binary: use positive class)
    if len(CLASS_NAMES) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = results_dir / "roc_curve.png"
        plt.savefig(roc_path, dpi=100)
        plt.close()
        print(f"Wrote {roc_path}")

    return report_path
