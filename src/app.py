"""
Dogs vs Cats image classifier — EfficientNet-B0 transfer learning.
Mirrors the improved pipeline in explore.ipynb.
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW_TRAIN_FLAT = ROOT / "data" / "raw" / "dogs-vs-cats" / "dogs-vs-cats" / "train"
TRAIN_DIR = ROOT / "data" / "raw" / "train"
TEST_DIR = ROOT / "data" / "raw" / "test"
MODELS_DIR = ROOT / "models"
MODEL_INPUT = (224, 224, 3)
BATCH_SIZE = 32


def ensure_class_split(src_dir: Path, train_dir: Path, test_dir: Path, test_ratio: float = 0.2) -> None:
    already = (train_dir / "dog").exists() and any((train_dir / "dog").glob("*.jpg"))
    if already:
        return
    for root in (train_dir, test_dir):
        for cls in ("dog", "cat"):
            (root / cls).mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    for cls in ("dog", "cat"):
        files = sorted(src_dir.glob(f"{cls}.*.jpg"))
        rng.shuffle(files)
        n_test = int(len(files) * test_ratio)
        for f in files[n_test:]:
            dst = train_dir / cls / f.name
            if not dst.exists():
                try:
                    os.link(f, dst)
                except OSError:
                    shutil.copy2(f, dst)
        for f in files[:n_test]:
            dst = test_dir / cls / f.name
            if not dst.exists():
                try:
                    os.link(f, dst)
                except OSError:
                    shutil.copy2(f, dst)


def build_model() -> tuple[Sequential, EfficientNetB0]:
    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=MODEL_INPUT)
    base.trainable = False
    model = Sequential(
        [
            base,
            GlobalAveragePooling2D(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(2, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_class_split(RAW_TRAIN_FLAT, TRAIN_DIR, TEST_DIR)

    target = (MODEL_INPUT[0], MODEL_INPUT[1])
    trdata = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    tsdata = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_gen = trdata.flow_from_directory(
        TRAIN_DIR,
        target_size=target,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=["dog", "cat"],
        shuffle=True,
        seed=SEED,
    )
    test_gen = tsdata.flow_from_directory(
        TEST_DIR,
        target_size=target,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=["dog", "cat"],
        shuffle=False,
    )

    model, base = build_model()
    model.fit(
        train_gen,
        steps_per_epoch=100,
        epochs=3,
        validation_data=test_gen,
        validation_steps=30,
    )

    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    best_path = MODELS_DIR / "efficientnet_b0_best.keras"
    callbacks = [
        ModelCheckpoint(str(best_path), monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=2, mode="max", restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1),
    ]
    model.fit(
        train_gen,
        steps_per_epoch=100,
        epochs=4,
        validation_data=test_gen,
        validation_steps=30,
        callbacks=callbacks,
    )

    best = load_model(best_path) if best_path.exists() else model
    loss, acc = best.evaluate(test_gen)
    print(f"Test loss={loss:.4f} accuracy={acc:.4f}")

    final_path = MODELS_DIR / "efficientnet_b0_dogs_cats.keras"
    best.save(final_path)
    print(f"Saved -> {final_path}")


if __name__ == "__main__":
    main()
