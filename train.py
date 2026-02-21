"""Training script for malaria detection models."""
import argparse
from pathlib import Path
from tensorflow import keras

from config import MODELS_DIR, RAW_DATA_DIR, IMG_SIZE, BATCH_SIZE, VAL_SPLIT, SEED
from data_loader import get_keras_image_dataset
from models import get_model


def main():
    parser = argparse.ArgumentParser(description="Train malaria detection model")
    parser.add_argument("--model", type=str, default="mobilenetv2", choices=["custom", "mobilenetv2", "efficientnet"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    args = parser.parse_args()

    train_ds, val_ds, _ = get_keras_image_dataset(
        args.data_dir, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, seed=SEED
    )
    if train_ds is None:
        print("No data found. Run download_data.py first or set --data-dir.")
        return 1

    model = get_model(args.model)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    save_dir = MODELS_DIR / args.model
    save_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(save_dir / "best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(save_dir / "final.keras")
    print(f"Saved to {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
