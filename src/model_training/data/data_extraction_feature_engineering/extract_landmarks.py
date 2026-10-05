"""Run MediaPipe HandLandmarker over a manifest and write a 63-feature CSV.
 
Output columns: split, label, image_path, handedness, x0,y0,z0 ... x20,y20,z20
Rows where extraction fails (no hand / multiple hands / unreadable) go to a
separate failures CSV so success/failure is tracked (report 3.3.2.3, step 4).
 
Raw MediaPipe values are stored; normalization is applied later in training so
the scaling choice can be changed without re-running extraction.
 
Requires: pip install mediapipe opencv-python numpy
Model file: download hand_landmarker.task from the MediaPipe docs
(https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) - verify the URL there.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

LOGGER = logging.getLogger(__name__)
NUM_LANDMARKS = 21
MAX_SIDE = 1280 # downscale FullHD images abit to speed up extraction

def feature_header() -> List[str]:
    return [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")]

def build_landmarker(model_path: Path) -> vision.HandLandmarker:
    if not model_path.is_file():
        raise RuntimeError(f"Model file not found: {model_path}")
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2, # as for 2 so we can reject multi hand images
        min_hand_detection_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)

def load_rgb(path: Path) -> Optional[np.ndarray]:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    height, width = bgr.shape[:2]
    scale = MAX_SIDE / max(height, width)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(width*scale), int(height*scale)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    
def extract_one(landmarker: vision.HandLandmarker, path: Path) -> tuple[Optional[List[float]], str, str]:
    # Returns (features, handedness, failure_reason). Feature is None on failure
    rgb = load_rgb(path)
    
    if rgb is None:
        return None, "", "unreadable_image"
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect(image)
    if len(results.hand_landmarks) == 0:
        return None, "", "no_hand"
    if len(results.hand_landmarks) > 1:
        return None, "", "multiple_hands"
    features = [v for lm in results.hand_landmarks[0] for v in (lm.x, lm.y, lm.z)]
    handedness = results.handedness[0][0].category_name if results.handedness else ""
    return features, handedness, ""

def run(manifest: Path, model_path: Path, out_csv: Path, failures_csv:Path) -> tuple[int, int]:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ok = failed = 0
    landmarker = build_landmarker(model_path)
    try:
        with manifest.open("r", newline="", encoding="utf-8") as src, \
                out_csv.open("w", newline="", encoding="utf-8") as dst, \
                failures_csv.open("w", newline="", encoding="utf-8") as fail:
            reader = csv.DictReader(src)
            writer = csv.writer(dst)
            fail_writer = csv.writer(fail)
            writer.writerow(["split", "label", "image_path", "handedness", *feature_header()])
            fail_writer.writerow(["split", "label", "image_path", "reason"])
            for row in reader:
                features, handedness, reason = extract_one(landmarker, Path(row["image_path"]))
                if features is None:
                    fail_writer.writerow([row["split"], row["label"], row["image_path"], reason])
                    failed += 1
                    continue
                writer.writerow([row["split"], row["label"], row["image_path"], handedness, *features])
                ok+=1
                if ok % 500 == 0:
                    LOGGER.info("extracted %d (failed %d)", ok, failed)
    except OSError as exc:
        raise RuntimeError(f"File error during extraction: {exc}") from exc
    finally:
        landmarker.close()
    return ok, failed

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--model", type=Path, required=True, help="path to hand_landmarker.task")
    parser.add_argument("--out", type=Path, default=Path("data/landmarks.csv"))
    parser.add_argument("--failures", type=Path, default=Path("data/landmark_failures.csv"))
    return parser.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        ok, failed = run(args.manifest, args.model, args.out, args.failures)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("done: %d ok, %d failed (%.1f%% success)", ok, failed, 100.0* ok / max(ok + failed, 1))
    return 0

if __name__ == "__main__":
    sys.exit(main())