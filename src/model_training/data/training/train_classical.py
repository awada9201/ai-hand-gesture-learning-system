"""Train KNN, SVM and Random Forest on normalised landmarks.
 
Hyper-parameters are tuned with GridSearchCV using the manifest's validation
split as the single validation fold. The winning configuration is then refit on
train+val and scored once on the untouched test split.
 
Usage:
    python -m src.models.train_classical --landmarks data/landmarks.csv --out-dir models
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import build_label_map, encode_labels, load_splits, save_label_map

LOGGER = logging.getLogger(__name__)
SEED = 42

def model_grids(seed: int = SEED) -> Dict[str, Tuple[BaseEstimator, Dict[str, List[Any]]]]:
    return {
        "knn": (
            Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [3, 5, 7, 11], "clf__weights": ["uniform", "distance"]},
        ),
        "svm":(
            Pipeline([("scale", StandardScaler()), ("clf", SVC(random_state=seed))]),
            {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1]}
        ),
        "rf": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {"n_estimators": [100, 300], "max_depth": [None, 20]},
        )
    }

def tune_and_evaluate(
    name: str,
    estimator: BaseEstimator,
    grid: Dict[str, List[Any]],
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    classes: Sequence[str],
) -> Tuple[BaseEstimator, Dict[str, Any], str]:
    x_train, y_train = data["train"]
    x_val, y_val = data["val"]
    x_test, y_test = data["test"]
    
    x_all = np.vstack([x_train, x_val])
    y_all = np.concatenate([y_train, y_val])
    
    fold = np.concatenate([np.full(len(y_train), -1), np.zeros(len(y_val), dtype=int)])
    
    search = GridSearchCV(
        estimator, grid, cv=PredefinedSplit(fold), scoring="accuracy", refit=True, n_jobs=1
    )
    
    start = time.time()
    search.fit(x_all, y_all)
    elapsed = time.time() - start
    
    best = search.best_estimator_
    start = time.time()
    pred = best.predict(x_test)
    latency_ms = 1000.0*(time.time() - start) / max(len(y_test), 1)
    
    metrics: Dict[str, Any] = {
        "best_params": search.best_params_,
        "val_accuracy": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_macro_f1": float(f1_score(y_test, pred, average="macro")),
        "fit_seconds": round(elapsed, 1),
        "predict_ms_per_sample": round(latency_ms, 4),
    }
    
    report = classification_report(y_test, pred, target_names=list(classes), digits=4)
    
    LOGGER.info(
        "%s: val=%.4f test=%.4f f1=%.4f params=%s",
        name, metrics["val_accuracy"], metrics["test_accuracy"], metrics["test_macro_f1"], search.best_params_,
    )
    return best, metrics, report

def run(landmarks: Path, out_dir: Path, only: Optional[Sequence[str]]) -> Dict[str, Any]:
    splits = load_splits(landmarks)
    classes = build_label_map(splits)
    data = {k: (x, encode_labels(y, classes)) for k, (x, y) in splits.items()}
    LOGGER.info(
        "classes=%d train=%d val=%d test=%d", len(classes),
        len(data["train"][1]), len(data["val"][1]), len(data["test"][1]),
    )
 
    out_dir.mkdir(parents=True, exist_ok=True)
    save_label_map(classes, out_dir / "label_map.json")
    all_metrics: Dict[str, Any] = {}
    for name, (estimator, grid) in model_grids().items():
        if only and name not in only:
            continue
        model, metrics, report = tune_and_evaluate(name, estimator, grid, data, classes)
        try:
            joblib.dump(model, out_dir / f"{name}.joblib")
            (out_dir / f"report_{name}.txt").write_text(report, encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Could not save {name} outputs: {exc}") from exc
        all_metrics[name] = metrics
 
    try:
        (out_dir / "metrics_classical.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not write metrics: {exc}") from exc
    return all_metrics

def parse_args(parv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landmarks", type=Path, default=Path("data/landmarks.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--only", nargs="*", choices=["knn", "svm", "rf"], help="train a subset")
    return parser.parse_args(parv)

def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        run(args.landmarks, args.out_dir, args.only)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
    