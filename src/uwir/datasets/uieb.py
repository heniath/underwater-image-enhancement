"""Strict UIEB discovery and the project-established 800/T90 split."""

from __future__ import annotations

import json
from pathlib import Path

from .common import PairedImageDataset, PairRecord, image_files, index_unique, seeded_partition

SPLIT_SEED = 42


def discover_uieb(root: str | Path) -> list[PairRecord]:
    root = Path(root).expanduser().resolve()
    input_root, target_root = root / "raw-890", root / "reference-890"
    if not input_root.is_dir() or not target_root.is_dir():
        raise FileNotFoundError(f"UIEB requires {input_root} and {target_root}")
    inputs = index_unique(image_files(input_root), input_root)
    targets = index_unique(image_files(target_root), target_root)
    unmatched_inputs = sorted(inputs.keys() - targets.keys())
    unmatched_targets = sorted(targets.keys() - inputs.keys())
    if unmatched_inputs or unmatched_targets:
        raise ValueError(
            f"UIEB pairing failed: unmatched inputs={unmatched_inputs[:10]}, "
            f"unmatched targets={unmatched_targets[:10]}"
        )
    records = [
        PairRecord(identity, str(inputs[identity]), str(targets[identity]))
        for identity in sorted(inputs)
    ]
    if len(records) != 890:
        raise ValueError(f"UIEB must contain exactly 890 paired images, found {len(records)}")
    return records


def _manifest(records: list[PairRecord]) -> dict[str, object]:
    development, test = records[:800], records[800:]
    train, validation = seeded_partition(development, (720, 80), SPLIT_SEED)
    return {
        "dataset": "UIEB",
        "split_source": "project_established_first_800_last_90_sorted_identity",
        "split_seed": SPLIT_SEED,
        "total_pairs": 890,
        "development_pairs": 800,
        "train_pairs": 720,
        "val_pairs": 80,
        "test_pairs": 90,
        "train": [record.json() for record in train],
        "validation": [record.json() for record in validation],
        "test": [record.json() for record in test],
    }


def prepare_uieb_splits(root: str | Path, manifest_path: str | Path) -> dict[str, list[PairRecord]]:
    records = discover_uieb(root)
    expected = _manifest(records)
    path = Path(manifest_path)
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != expected:
            raise ValueError(f"Frozen UIEB manifest differs from discovered dataset: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return {
        key: [PairRecord(**item) for item in expected[name]]
        for key, name in (("train", "train"), ("val", "validation"), ("test", "test"))
    }


def build_uieb_datasets(
    root: str | Path, manifest_path: str | Path, *, crop_size=256, cache="auto"
):
    splits = prepare_uieb_splits(root, manifest_path)
    return {
        name: PairedImageDataset(
            records, training=name == "train", crop_size=crop_size, cache=cache
        )
        for name, records in splits.items()
    }
