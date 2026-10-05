"""Load landmarks.csv and normalise hand landmarks for training and inference.
 
The Kotlin app must apply EXACTLY the same normalisation before calling the
exported model (see normalize_landmarks).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3
FEATURE_COLUMNS: List[str] = [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")]
SPLITS: Tuple[str, ...] = ("train", "val", "test")
_EPS = 1e-6

def normalize_landmarks(raw: np.ndarray) -> np.ndarray:
    """Make landmarks independent of hand position and hand size.
 
    1. Subtract the wrist (landmark 0) from every point -> translation invariant.
    2. Divide by the largest wrist-to-landmark distance -> scale invariant.
 
    Args:
        raw: array of shape (N, 63) in MediaPipe order x0,y0,z0,...,x20,y20,z20.
    Returns:
        array of shape (N, 63), float32.
    """
    if raw.ndim != 2 or raw.shape[1] != NUM_FEATURES:
        raise ValueError(f"Expected shape (N, {NUM_FEATURES}), got {raw.shape}")
    points = raw.reshape(-1, NUM_LANDMARKS, 3).astype(np.float32)
    points = points - points[:, 0:1, :]
    scale = np.linalg.norm(points, axis=2).max(axis=1, keepdims=True) # (N, 1)
    points = points / np.maximum(scale[:, :, None], _EPS)
    return points.reshape(-1, NUM_FEATURES)
    
def load_splits(csv_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Read landmarks.csv and return {split: (normalised_features, string_labels)}."""
    try:
        frame = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise RuntimeError(f"Could not read {csv_path}: {exc}") from exc
 
    missing = [c for c in ["split", "label", *FEATURE_COLUMNS] if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{csv_path} is missing columns: {missing[:5]}...")
 
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for split in SPLITS:
        part = frame[frame["split"] == split].dropna(subset=FEATURE_COLUMNS)
        if part.empty:
            raise RuntimeError(f"No rows for split '{split}' in {csv_path}")
        features = normalize_landmarks(part[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        out[split] = (features, part["label"].to_numpy(dtype=str))
    return out
 
 
def build_label_map(splits: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> List[str]:
    """Sorted class names; the index in this list is the integer label."""
    return sorted({label for _, labels in splits.values() for label in labels})
 
 
def encode_labels(labels: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(classes)}
    try:
        return np.array([index[label] for label in labels], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"Unknown class label: {exc}") from exc
 
 
def save_label_map(classes: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(list(classes), indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not write {path}: {exc}") from exc