"""Shared paired-image primitives; dataset-specific discovery lives beside this file."""

from __future__ import annotations

import copy
import os
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PairRecord:
    identity: str
    input: str
    target: str
    split_hint: str | None = None

    def json(self) -> dict[str, str | None]:
        return asdict(self)


def image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def index_unique(paths: Iterable[Path], root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}
    for path in paths:
        identity = path.stem.casefold()
        if identity in result:
            duplicates.setdefault(identity, [str(result[identity].relative_to(root))]).append(
                str(path.relative_to(root))
            )
        else:
            result[identity] = path
    if duplicates:
        preview = dict(list(duplicates.items())[:10])
        raise ValueError(f"Duplicate image identities make pairing ambiguous: {preview}")
    return result


def available_memory_bytes() -> int | None:
    try:
        values = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        line = next(item for item in values if item.startswith("MemAvailable:"))
        return int(line.split()[1]) * 1024
    except (OSError, StopIteration, ValueError):
        return None


def decoded_size_estimate(records: Iterable[PairRecord]) -> int:
    total = 0
    for record in records:
        for filename in (record.input, record.target):
            with Image.open(filename) as image:
                total += image.width * image.height * 3
    return total


class PairedImageDataset(Dataset):
    """RGB paired dataset with aligned spatial transforms and optional RAM cache."""

    def __init__(
        self,
        records: Iterable[PairRecord],
        *,
        training: bool,
        crop_size: int = 256,
        cache: bool | str = "auto",
        cache_fraction: float = 0.5,
    ) -> None:
        self.records = tuple(records)
        self.training = training
        self.crop_size = crop_size
        self.cache_requested = cache
        self._cache: list[tuple[Image.Image, Image.Image]] | None = None
        should_cache = bool(cache)
        if cache == "auto":
            available = available_memory_bytes()
            estimate = decoded_size_estimate(self.records)
            should_cache = available is not None and estimate <= available * cache_fraction
        if should_cache:
            self._cache = [self._load_pair(record) for record in self.records]

    @staticmethod
    def _load_pair(record: PairRecord) -> tuple[Image.Image, Image.Image]:
        with Image.open(record.input) as source:
            degraded = source.convert("RGB").copy()
        with Image.open(record.target) as source:
            reference = source.convert("RGB").copy()
        if degraded.size != reference.size:
            raise ValueError(
                f"Pair {record.identity!r} is spatially misaligned: "
                f"input={degraded.size}, target={reference.size}"
            )
        return degraded, reference

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        if self._cache is None:
            degraded, reference = self._load_pair(record)
        else:
            degraded, reference = (image.copy() for image in self._cache[index])

        size = self.crop_size
        if min(degraded.size) < size:
            scale = size / min(degraded.size)
            resized = (round(degraded.height * scale), round(degraded.width * scale))
            degraded = TF.resize(degraded, resized, antialias=True)
            reference = TF.resize(reference, resized, antialias=True)
        if self.training:
            top, left, height, width = torch.randint(0, 2**31 - 1, (1,)).item(), 0, size, size
            max_top = degraded.height - size
            max_left = degraded.width - size
            top = top % (max_top + 1)
            left = torch.randint(0, max_left + 1, (1,)).item()
            degraded = TF.crop(degraded, top, left, height, width)
            reference = TF.crop(reference, top, left, height, width)
            if torch.rand(()) < 0.5:
                degraded, reference = TF.hflip(degraded), TF.hflip(reference)
            if torch.rand(()) < 0.5:
                degraded, reference = TF.vflip(degraded), TF.vflip(reference)
        return {
            "degraded": TF.to_tensor(degraded),
            "reference": TF.to_tensor(reference),
            "identity": record.identity,
            "input_path": record.input,
            "target_path": record.target,
        }


def clone_for_split(dataset: PairedImageDataset, records: Iterable[PairRecord], training: bool):
    clone = copy.copy(dataset)
    clone.records = tuple(records)
    clone.training = training
    clone._cache = None
    return clone


def seeded_partition(
    records: list[PairRecord], sizes: tuple[int, ...], seed: int
) -> list[list[PairRecord]]:
    if sum(sizes) != len(records):
        raise ValueError(f"Split sizes {sizes} do not cover {len(records)} records")
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    result, offset = [], 0
    for size in sizes:
        result.append([records[index] for index in indices[offset : offset + size]])
        offset += size
    return result


def resolve_data_root(data_root: str | os.PathLike[str] | None = None) -> Path:
    if data_root is not None:
        return Path(data_root).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "datasets"
