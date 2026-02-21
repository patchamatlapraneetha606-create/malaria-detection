"""Custom CNN and transfer learning models (MobileNetV2, EfficientNetB0)."""
from tensorflow import keras
from tensorflow.keras import layers

from config import IMG_SIZE, NUM_CLASSES


def build_custom_cnn(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """Custom CNN: Conv2D + BN + Dropout."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def build_mobilenetv2(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """MobileNetV2 transfer learning with ImageNet weights (falls back to random if download fails)."""
    try:
        base = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
    except Exception:
        base = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=None,
            pooling="avg",
        )
    base.trainable = False
    inputs = keras.Input(shape=input_shape)
    x = base(inputs)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def build_efficientnet(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """EfficientNetB0 transfer learning (falls back to random if ImageNet download fails)."""
    try:
        base = keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
    except Exception:
        base = keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights=None,
            pooling="avg",
        )
    base.trainable = False
    inputs = keras.Input(shape=input_shape)
    x = base(inputs)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def get_model(name: str):
    """Return model by name: custom, mobilenetv2, efficientnet."""
    name = (name or "mobilenetv2").lower()
    if name == "custom":
        return build_custom_cnn()
    if name == "mobilenetv2":
        return build_mobilenetv2()
    if name == "efficientnet":
        return build_efficientnet()
    raise ValueError(f"Unknown model: {name}")
