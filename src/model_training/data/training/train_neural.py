"""Train a Keras network on normalised landmarks and export it to TensorFlow Lite.
 
Architectures:
  mlp  - Dense network on the 63 flattened features (recommended for the app).
  lstm - treats the 21 landmarks as a 21-step sequence of (x, y, z). HaGRID is
         static images, so this is NOT a temporal model; it only lets you
         report an LSTM. Real dynamic-gesture LSTMs need frame sequences.
 
Usage:
    python -m src.models.train_neural --landmarks data/landmarks.csv --arch mlp
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

from sklearn.metrics import accuracy_score, classification_report, f1_score
from .features import (
    NUM_FEATURES, NUM_LANDMARKS, build_label_map, encode_labels, load_splits, save_label_map,
)

LOGGER = logging.getLogger(__name__)
SEED = 42

def build_model(num_classes: int) -> tf.keras.Model:
    # Multi-Layer Perceptron
    layers = [
        tf.keras.layers.Input(shape=(NUM_FEATURES,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def export_tflite(model: tf.keras.Model, path:Path) -> None:
    """Convert to a float32 .tflite file (no quantisation, so accuracy is unchanged)"""
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(converter.convert())
    except (ValueError, RuntimeError, OSError) as exc:
        raise RuntimeError(f"TFLite export failed: {exc}")
    
def tflite_predict(path: Path, features: np.ndarray) -> np.ndarray:
    """Run the exported .tflite file one sample at a time and return class ids."""
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    preds = np.empty(len(features), dtype=np.int64)
    for i, row in enumerate(features):
        interpreter.set_tensor(inp["index"], row[None, :].astype(np.float32))
        interpreter.invoke()
        preds[i] = int(np.argmax(interpreter.get_tensor(out["index"])[0]))
    return preds
 
 
def run(landmarks: Path, out_dir: Path, epochs: int, batch_size: int) -> Dict[str, Any]:
    tf.keras.utils.set_random_seed(SEED)
    splits = load_splits(landmarks)
    classes = build_label_map(splits)
    x_train, y_train = splits["train"][0], encode_labels(splits["train"][1], classes)
    x_val, y_val = splits["val"][0], encode_labels(splits["val"][1], classes)
    x_test, y_test = splits["test"][0], encode_labels(splits["test"][1], classes)
 
    model = build_model(len(classes))
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size, callbacks=[stop], verbose=2,
    )
 
    keras_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    tflite_path = out_dir / "mcp.tflite"
    export_tflite(model, tflite_path)
    lite_pred = tflite_predict(tflite_path, x_test)
 
    out_dir.mkdir(parents=True, exist_ok=True)
    save_label_map(classes, out_dir / "label_map.json")
    metrics: Dict[str, Any] = {
        "epochs_run": len(history.history["loss"]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "test_accuracy_keras": float(accuracy_score(y_test, keras_pred)),
        "test_accuracy_tflite": float(accuracy_score(y_test, lite_pred)),
        "test_macro_f1_keras": float(f1_score(y_test, keras_pred, average="macro")),
        "keras_tflite_agreement": float(np.mean(keras_pred == lite_pred)),
        "tflite_size_kb": round(tflite_path.stat().st_size / 1024, 1),
    }
    try:
        model.save(out_dir / f"mlp.keras")
        (out_dir / f"metrics_mlp.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (out_dir / f"report_mlp.txt").write_text(
            classification_report(y_test, keras_pred, target_names=list(classes), digits=4), encoding="utf-8"
        )
    except OSError as exc:
        raise RuntimeError(f"Could not save outputs: {exc}") from exc
    LOGGER.info("%s: %s", "mlp", metrics)
    return metrics
 
 
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landmarks", type=Path, default=Path("data/landmarks.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)
 
 
def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        run(args.landmarks, args.out_dir, args.epochs, args.batch_size)
    except (RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())