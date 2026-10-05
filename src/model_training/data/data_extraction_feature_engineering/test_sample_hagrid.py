"""Tests for src/data/sample_hagrid.py"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

import sample_hagrid as sh


def _make_tree(root: Path, labels: "list[str]", n: int, nested: bool = False) -> None:
    for label in labels:
        folder = root / "train_val" / label if nested else root / label
        folder.mkdir(parents=True, exist_ok=True)
        for k in range(n):
            (folder / f"{label}_{k}.jpg").write_bytes(b"x")


def test_split_quotas_sum() -> None:
    assert sum(sh.split_quotas(101).values()) == 101


@pytest.mark.parametrize("nested", [False, True])
def test_balanced_split(tmp_path: Path, nested: bool) -> None:
    _make_tree(tmp_path, ["fist", "palm", "like"], 300, nested)
    rows = sh.build_manifest(sh.scan_images(tmp_path), ["fist", "palm"], 100, 1)
    counts = Counter((split, label) for split, label, _ in rows)
    for label in ("fist", "palm"):
        assert (counts[("train", label)], counts[("val", label)], counts[("test", label)]) == (70, 15, 15)
    assert all("like" not in path for _, _, path in rows)
    assert len({p for _, _, p in rows}) == len(rows)


def test_missing_class_raises(tmp_path: Path) -> None:
    _make_tree(tmp_path, ["fist"], 10)
    with pytest.raises(RuntimeError):
        sh.build_manifest(sh.scan_images(tmp_path), ["fist", "ghost"], 5, 1)


def test_seed_reproducible(tmp_path: Path) -> None:
    _make_tree(tmp_path, ["fist"], 200)
    grouped = sh.scan_images(tmp_path)
    assert sh.build_manifest(grouped, ["fist"], 50, 7) == sh.build_manifest(grouped, ["fist"], 50, 7)