"""Обучение модели из data/train (подпапки = классы)."""

import argparse
import json
from pathlib import Path

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import image_dataset_from_directory

from model import build_emotion_cnn

DATA_DIR = Path(__file__).resolve().parent / "data" / "train"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.keras"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

IMG_SIZE = 48
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

DS_KWARGS = {
    "labels": "inferred",
    "label_mode": "int",
    "color_mode": "grayscale",
    "batch_size": BATCH_SIZE,
    "image_size": (IMG_SIZE, IMG_SIZE),
    "validation_split": VALIDATION_SPLIT,
    "seed": 42,
}


REDUCE_LR = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    verbose=0,
    min_lr=1e-6,
)
EARLY_STOPPING = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
)


def load_datasets(data_dir: Path):
    train_ds = image_dataset_from_directory(data_dir, subset="training", **DS_KWARGS)
    val_ds = image_dataset_from_directory(data_dir, subset="validation", **DS_KWARGS)
    return train_ds, val_ds


def save_artifacts(model, class_names: list[str], epochs: int, history) -> float:
    model.save(MODEL_PATH)
    val_acc = float(max(history.history["val_accuracy"]))
    METADATA_PATH.write_text(
        json.dumps(
            {
                "classes": class_names,
                "img_size": IMG_SIZE,
                "epochs": epochs,
                "val_accuracy": val_acc,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return val_acc


def compute_class_weight(class_names: list[str], data_dir: Path) -> dict[int, float]:
    counts = [
        sum(1 for p in (data_dir / name).iterdir() if p.is_file())
        for name in class_names
    ]
    total = sum(counts)
    n = len(class_names)
    return {i: total / (n * count) for i, count in enumerate(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение модели эмоций")
    parser.add_argument("--epochs", type=int, default=10, help="Количество эпох")
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = load_datasets(DATA_DIR)
    class_names = train_ds.class_names  # pyright: ignore[reportAttributeAccessIssue]

    class_weight = compute_class_weight(class_names, DATA_DIR)
    for name, idx in zip(class_names, range(len(class_names))):
        print(f"class_weight[{name}] = {class_weight[idx]:.4f}")

    model = build_emotion_cnn((IMG_SIZE, IMG_SIZE, 1), len(class_names))
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=[REDUCE_LR, EARLY_STOPPING],
    )

    val_acc = save_artifacts(model, class_names, args.epochs, history)

    print(f"Модель: {MODEL_PATH}")
    print(f"Классы: {', '.join(class_names)}")
    print(f"Val accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
