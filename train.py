from pathlib import Path
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset"
MODEL_DIR = BASE / "model"
RESULTS = BASE / "results"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

# Implementation choice. The PPT does not specify an exact split ratio.
TEST_RATIO = 0.15
VAL_RATIO = 0.15

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_DIR.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def discover_classes():
    folders = []
    for p in DATASET.rglob("*"):
        if p.is_dir():
            images = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in EXTENSIONS]
            if images:
                folders.append(p)

    if not folders:
        raise FileNotFoundError(
            "No class folders containing images were found in dataset/."
        )

    # Keep deepest image-containing folders so crop grouping folders are not treated as classes.
    folders = sorted(folders, key=lambda x: len(x.parts), reverse=True)
    selected = []
    for p in folders:
        if not any(parent in p.parents for parent in selected):
            selected.append(p)

    return sorted(selected, key=lambda x: str(x).lower())

def collect():
    class_dirs = discover_classes()
    class_names = [p.name for p in class_dirs]
    samples = []
    for label, folder in enumerate(class_dirs):
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in EXTENSIONS:
                samples.append((str(p), label))
    return samples, class_names

def split_samples(samples):
    groups = {}
    for path, label in samples:
        groups.setdefault(label, []).append(path)

    rng = random.Random(SEED)
    train, val, test = [], [], []

    for label, paths in sorted(groups.items()):
        rng.shuffle(paths)
        n = len(paths)
        n_test = max(1, int(round(n * TEST_RATIO)))
        n_val = max(1, int(round(n * VAL_RATIO)))
        if n_test + n_val >= n:
            n_test, n_val = 1, 1

        test += [(p, label) for p in paths[:n_test]]
        val += [(p, label) for p in paths[n_test:n_test+n_val]]
        train += [(p, label) for p in paths[n_test+n_val:]]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test

def make_dataset(items, training=False):
    paths = [x[0] for x in items]
    labels = [x[1] for x in items]

    def gen():
        indices = list(range(len(paths)))
        if training:
            random.shuffle(indices)
        for i in indices:
            try:
                raw = tf.io.read_file(paths[i])
                img = tf.image.decode_image(raw, channels=3, expand_animations=False)
                img = tf.image.resize(img, IMAGE_SIZE)
                yield img, np.int32(labels[i])
            except Exception as e:
                print("Skipping unreadable image:", paths[i], e)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((*IMAGE_SIZE, 3), tf.float32),
            tf.TensorSpec((), tf.int32)
        )
    )

    if training:
        aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10)
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(lambda x, y: (preprocess_input(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def build_model(num_classes):
    base = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3)
    )
    base.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    if not DATASET.exists():
        raise FileNotFoundError(
            "dataset/ was not found. Put your downloaded Kaggle dataset inside dataset/."
        )

    samples, classes = collect()
    print(f"Found {len(samples)} images across {len(classes)} classes.")
    for i, c in enumerate(classes):
        print(f"{i}: {c} ({sum(1 for _, y in samples if y == i)})")

    train_items, val_items, test_items = split_samples(samples)
    print(f"Train: {len(train_items)} | Validation: {len(val_items)} | Test: {len(test_items)}")

    train_ds = make_dataset(train_items, True)
    val_ds = make_dataset(val_items)
    test_ds = make_dataset(test_items)

    model = build_model(len(classes))
    model_path = MODEL_DIR / "crop_disease_model.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path), monitor="val_accuracy",
            save_best_only=True, mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model = tf.keras.models.load_model(model_path)
    y_true, y_pred = [], []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "total_images_found": len(samples),
        "num_classes": len(classes),
        "class_names": classes,
        "train_images": len(train_items),
        "validation_images": len(val_items),
        "test_images": len(test_items),
        "split_note": "Implementation choice: 15% test, 15% validation, remaining training.",
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(classes)))
        ).tolist()
    }

    (MODEL_DIR / "class_names.json").write_text(
        json.dumps(classes, indent=2), encoding="utf-8"
    )
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    summary = {
        "total_images": len(samples),
        "classes": {
            classes[i]: sum(1 for _, y in samples if y == i)
            for i in range(len(classes))
        }
    }
    (MODEL_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "accuracy_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "loss_curve.png", dpi=150)
    plt.close()

    print("\nTraining complete.")
    print(f"Model: {model_path}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    main()
