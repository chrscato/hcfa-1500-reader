"""Deterministic train/val/test splits, stratified by tier.

Reads the synth manifest (`data/<batch>/manifest.jsonl`), partitions each
tier independently with a seeded shuffle, then writes:

    splits/<batch>/train.jsonl
    splits/<batch>/val.jsonl
    splits/<batch>/test.jsonl

Each line is the original manifest row plus a `split` field. Paths in the
manifest are kept relative to the batch dir (so they survive being moved).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _read_manifest(path: Path) -> List[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def make_splits(
    rows: List[dict],
    *,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 0,
) -> Dict[str, List[dict]]:
    """Stratify by tier; within each tier, shuffle deterministically and slice.

    Slicing uses round-to-nearest with a guarantee of at least 1 per non-empty
    bucket as long as there are >=3 rows in the tier — otherwise the smallest
    buckets may be empty.
    """
    by_tier: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_tier[r["tier"]].append(r)

    rng = random.Random(seed)
    splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for tier, items in sorted(by_tier.items()):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_val = max(1, round(n * val_frac)) if n >= 3 else 0
        n_test = max(1, round(n * test_frac)) if n >= 3 else 0
        n_train = n - n_val - n_test
        if n_train < 0:
            n_train, n_val, n_test = n, 0, 0
        train = items[:n_train]
        val = items[n_train : n_train + n_val]
        test = items[n_train + n_val :]
        for r in train:
            splits["train"].append({**r, "split": "train"})
        for r in val:
            splits["val"].append({**r, "split": "val"})
        for r in test:
            splits["test"].append({**r, "split": "test"})
    return splits


def write_splits(splits: Dict[str, List[dict]], out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for name, rows in splits.items():
        p = out_dir / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        paths[name] = p
    return paths


def split_manifest(manifest_path: Path, out_dir: Path, *, seed: int = 0) -> Tuple[Dict[str, Path], Dict[str, List[dict]]]:
    rows = _read_manifest(manifest_path)
    splits = make_splits(rows, seed=seed)
    paths = write_splits(splits, out_dir)
    return paths, splits
