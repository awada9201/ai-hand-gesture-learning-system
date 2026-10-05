"""Build a balanced random sample manifest from a local folder of HaGRID images.
 
Works on a downloaded sample (see download_sample.py). It does not need the
annotation files: the class is taken from the name of the folder the image sits
in. Images with more than one hand are rejected later by extract_landmarks.py.
 
Output is a CSV manifest (split, label, image_path). No images are copied.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

# Report, Table 6: 70 / 15 / 15
SPLIT_RATIOS: Tuple[Tuple[str, int], ...] = (("train", 70), ("val", 15), ("test", 15))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def label_from_path(path: Path, root: Path) -> Optional[str]:
    # Returns the nearest parent folder name (relative to root), or None
    try:
        parts = path.relative_to(root).parts[:-1]
    except ValueError:
        return None
    
    name = parts[-1]
    # Strip HaGRID's split prefixes such as "train_val_", "train_", "test_"
    for prefix in ("train_val_", "train_", "val_", "test_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name

def scan_images(root: Path) -> Dict[str, List[Path]]:
    # Group every image under 'root' by its parent folder name
    if not root.is_dir():
        raise RuntimeError(f"Images folder not found: {root}")
    grouped: Dict[str, List[Path]] = {}
    
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
            continue
        label = label_from_path(path, root)
        if label is not None:
            grouped.setdefault(label, []).append(path)
    return grouped

def split_quotas(total: int) -> Dict[str, int]:
    quotas = {name: total * pct // 100 for name, pct, in SPLIT_RATIOS}
    quotas[SPLIT_RATIOS[0][0]] += total - sum(quotas.values())  # Add remainder to first split
    return quotas

def build_manifest(
        grouped: Dict[str, List[Path]], 
        classes: Sequence[str], 
        per_class: int, 
        seed: int
    ) -> List[Tuple[str, str, str]]:
        missing = [c for c in classes if c not in grouped]
        if missing:
            found = ", ".join(f"{k} ({len(v)})" for k, v in sorted(grouped.items()))
            raise RuntimeError(f"Class folder(s) not found in {missing}. Folders found: {found}")

        rng = random.Random(seed)
        rows: List[Tuple[str, str, str]] = []
        for label in classes:
            paths = list(grouped[label])
            rng.shuffle(paths)
            if len(paths) < per_class:
                LOGGER.warning("%s: wanted %d, only %d available", label, per_class, len(paths))
            paths = paths[:per_class]
            quotas = split_quotas(len(paths))
            start = 0
            for split, _ in SPLIT_RATIOS:
                end = start + quotas[split]
                rows.extend((split, label, str(p)) for p in paths[start:end])
                start = end
        return rows
        
def write_manifest(rows: Sequence[Tuple[str, str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["split", "label", "image_path"])
            writer.writerows(rows)
    except OSError as exc:
        raise RuntimeError(f"Could not write manifest to {out_path}: {exc}") from exc
    
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--classes", nargs="*", default=[], help="your 8 classes folder names")
    parser.add_argument("--per-class", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--list-classes", action="store_true", help="print folders found and exit")
    return parser.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        grouped = scan_images(args.images_root)
        if args.list_classes or not args.classes:
            for name, count in sorted(Counter({k: len(v) for k, v in grouped.items()}).items()):
                print(f"{name}\t{count}")
            return 0
        rows = build_manifest(grouped, args.classes, args.per_class, args.seed)
        write_manifest(rows, args.out)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("Wrote %d rows to %s", len(rows), args.out)
    return 0

if __name__ == "__main__":
    sys.exit(main())