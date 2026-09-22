"""Conservative LSUI recursive discovery; ambiguous layouts are rejected."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .common import PairedImageDataset, PairRecord, image_files, index_unique, seeded_partition

SPLIT_SEED = 42
INPUT_TOKENS = {"input", "inputs", "inp", "degraded", "underwater", "raw"}
TARGET_TOKENS = {"gt", "gtr", "groundtruth", "ground_truth", "reference", "references", "target"}


def compact_tree(root: str | Path, max_entries: int = 80) -> str:
    root = Path(root)
    if not root.is_dir():
        return f"{root} [missing]"
    entries = sorted(root.rglob("*"))
    lines = [str(root)]
    for path in entries[:max_entries]:
        lines.append(f"  {path.relative_to(root)}{'/' if path.is_dir() else ''}")
    if len(entries) > max_entries:
        lines.append(f"  ... {len(entries) - max_entries} more entries")
    return "\n".join(lines)


def _role(path: Path, root: Path) -> str | None:
    tokens = {part.casefold().replace(" ", "_") for part in path.relative_to(root).parts[:-1]}
    input_hit, target_hit = bool(tokens & INPUT_TOKENS), bool(tokens & TARGET_TOKENS)
    if input_hit == target_hit:
        return None
    return "input" if input_hit else "target"


def _split_hint(path: Path, root: Path) -> str | None:
    tokens = {part.casefold() for part in path.relative_to(root).parts[:-1]}
    for name, aliases in (("train", {"train", "training"}), ("test", {"test", "testing"})):
        if tokens & aliases:
            return name
    return None


def _manifest_split_hints(root: Path) -> dict[str, str]:
    """Read explicit local train/test manifests before consulting directory names."""
    hints: dict[str, str] = {}

    def strings(value):
        values: list[str] = []
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, list):
                pending.extend(item)
            elif isinstance(item, dict):
                pending.extend(item.values())
        return values

    def register(values, split):
        for value in values:
            identity = Path(value.strip()).stem.casefold()
            if not identity:
                continue
            if identity in hints and hints[identity] != split:
                raise ValueError(f"LSUI manifests assign {identity!r} to both train and test")
            hints[identity] = split

    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in {".txt", ".csv", ".json"}
        and (
            {"train", "test", "manifest", "split"} & set(re.split(r"[^a-z]+", path.stem.casefold()))
        )
    ]
    for path in candidates:
        tokens = set(re.split(r"[^a-z]+", path.stem.casefold()))
        filename_split = (
            "train"
            if "train" in tokens and "test" not in tokens
            else "test"
            if "test" in tokens and "train" not in tokens
            else None
        )
        try:
            if path.suffix.casefold() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                if filename_split is not None:
                    register(strings(payload), filename_split)
                elif isinstance(payload, dict):
                    for key, value in payload.items():
                        key_tokens = set(re.split(r"[^a-z]+", str(key).casefold()))
                        if "train" in key_tokens:
                            register(strings(value), "train")
                        if "test" in key_tokens:
                            register(strings(value), "test")
            else:
                if filename_split is None:
                    continue
                values = re.split(r"[\s,]+", path.read_text(encoding="utf-8"))
                register(values, filename_split)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return hints


def discover_lsui(root: str | Path) -> tuple[list[PairRecord], dict[str, object]]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LSUI root does not exist: {root}")
    files = image_files(root)
    inputs = [path for path in files if _role(path, root) == "input"]
    targets = [path for path in files if _role(path, root) == "target"]
    unknown = [path for path in files if _role(path, root) is None]
    if not inputs or not targets:
        raise ValueError(
            "Could not identify LSUI input and GT directories from documented role names.\n"
            + compact_tree(root)
        )
    input_index, target_index = index_unique(inputs, root), index_unique(targets, root)
    manifest_hints = _manifest_split_hints(root)
    unmatched_inputs = sorted(input_index.keys() - target_index.keys())
    unmatched_targets = sorted(target_index.keys() - input_index.keys())
    if unmatched_inputs or unmatched_targets:
        raise ValueError(
            "LSUI exact-identity pairing is incomplete; refusing to zip sorted filenames. "
            f"unmatched_inputs={unmatched_inputs[:20]}, unmatched_gt={unmatched_targets[:20]}"
        )
    records = []
    for identity in sorted(input_index):
        input_hint = manifest_hints.get(identity, _split_hint(input_index[identity], root))
        target_hint = manifest_hints.get(identity, _split_hint(target_index[identity], root))
        if input_hint != target_hint:
            raise ValueError(
                f"LSUI split disagreement for {identity}: {input_hint} vs {target_hint}"
            )
        records.append(
            PairRecord(
                identity, str(input_index[identity]), str(target_index[identity]), input_hint
            )
        )
    report = {
        "input_count": len(inputs),
        "gt_count": len(targets),
        "matched_pairs": len(records),
        "unmatched_inputs": unmatched_inputs,
        "unmatched_gt": unmatched_targets,
        "duplicate_identities": [],
        "unclassified_image_count": len(unknown),
    }
    return records, report


def _build_manifest(records: list[PairRecord], report: dict[str, object]) -> dict[str, object]:
    official_train = [record for record in records if record.split_hint == "train"]
    official_test = [record for record in records if record.split_hint == "test"]
    unsplit = [record for record in records if record.split_hint is None]
    if official_train and official_test and not unsplit:
        val_size = max(1, round(len(official_train) * 0.1))
        train, val = seeded_partition(
            official_train, (len(official_train) - val_size, val_size), SPLIT_SEED
        )
        test, source = official_test, "official"
    elif not official_train and not official_test:
        test_size = max(1, round(len(records) * 0.1))
        val_size = max(1, round(len(records) * 0.1))
        train, val, test = seeded_partition(
            records, (len(records) - val_size - test_size, val_size, test_size), SPLIT_SEED
        )
        source = "project_defined"
    else:
        raise ValueError("LSUI contains a partial official split; refusing to invent membership")
    identities = [{record.identity for record in group} for group in (train, val, test)]
    if (
        identities[0] & identities[1]
        or identities[0] & identities[2]
        or identities[1] & identities[2]
    ):
        raise AssertionError("LSUI train/validation/test splits overlap")
    return {
        "dataset": "LSUI",
        "split_source": source,
        "split_seed": SPLIT_SEED,
        "total_pairs": len(records),
        "train_pairs": len(train),
        "val_pairs": len(val),
        "test_pairs": len(test),
        "discovery_report": report,
        "train": [record.json() for record in train],
        "validation": [record.json() for record in val],
        "test": [record.json() for record in test],
    }


def prepare_lsui_splits(root: str | Path, manifest_path: str | Path) -> dict[str, list[PairRecord]]:
    records, report = discover_lsui(root)
    expected = _build_manifest(records, report)
    path = Path(manifest_path)
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != expected:
            raise ValueError(f"Frozen LSUI manifest differs from discovered dataset: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return {
        key: [PairRecord(**item) for item in expected[name]]
        for key, name in (("train", "train"), ("val", "validation"), ("test", "test"))
    }


def build_lsui_datasets(
    root: str | Path, manifest_path: str | Path, *, crop_size=256, cache="auto"
):
    splits = prepare_lsui_splits(root, manifest_path)
    return {
        name: PairedImageDataset(
            records, training=name == "train", crop_size=crop_size, cache=cache
        )
        for name, records in splits.items()
    }
