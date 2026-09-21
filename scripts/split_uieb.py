"""
Split UIEB-890 into train (800) and test (90) subsets.

Creates the following layout:
    datasets/UIEB/train/input/       ← 800 degraded images (symlinks)
    datasets/UIEB/train/reference/   ← 800 GT images       (symlinks)
    datasets/UIEB/test/input/        ← 90 degraded images  (symlinks)
    datasets/UIEB/test/reference/    ← 90 GT images        (symlinks)

The original raw-890/ and reference-890/ directories are left untouched.

Usage:
    python scripts/split_uieb.py [--data_dir ./datasets/UIEB] [--seed 42] [--copy]

By default, files are **copied** (not symlinked) so the split works
portably on Windows without admin privileges.
"""

import argparse
import os
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Split UIEB-890 → 800 train + 90 test")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./datasets/UIEB",
        help="Root of the UIEB dataset (contains raw-890/ and reference-890/)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--n_test", type=int, default=90, help="Number of test images")
    parser.add_argument(
        "--symlink",
        action="store_true",
        default=False,
        help="Use symlinks instead of copying (requires admin on Windows)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    raw_dir = data_dir / "raw-890"
    ref_dir = data_dir / "reference-890"

    assert raw_dir.is_dir(), f"Missing: {raw_dir}"
    assert ref_dir.is_dir(), f"Missing: {ref_dir}"

    # Collect paired stems (only keep images present in BOTH dirs)
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    raw_stems = {f.stem: f for f in raw_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS}
    ref_stems = {f.stem: f for f in ref_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS}
    common_stems = sorted(set(raw_stems.keys()) & set(ref_stems.keys()))
    print(f"Found {len(common_stems)} paired images (raw & reference)")

    assert len(common_stems) >= args.n_test, (
        f"Need at least {args.n_test} paired images, found {len(common_stems)}"
    )

    # Deterministic random split
    rng = random.Random(args.seed)
    shuffled = list(common_stems)
    rng.shuffle(shuffled)
    test_stems = set(shuffled[: args.n_test])
    train_stems = set(shuffled[args.n_test :])

    print(f"Train: {len(train_stems)} | Test: {len(test_stems)}")

    # Create output directories
    splits = {
        "train": train_stems,
        "test": test_stems,
    }
    for split_name, stems in splits.items():
        input_dir = data_dir / split_name / "input"
        reference_dir = data_dir / split_name / "reference"
        input_dir.mkdir(parents=True, exist_ok=True)
        reference_dir.mkdir(parents=True, exist_ok=True)

        for stem in sorted(stems):
            src_raw = raw_stems[stem]
            src_ref = ref_stems[stem]
            dst_raw = input_dir / src_raw.name
            dst_ref = reference_dir / src_ref.name

            if args.symlink:
                # Symlink (may require admin on Windows)
                if not dst_raw.exists():
                    dst_raw.symlink_to(src_raw)
                if not dst_ref.exists():
                    dst_ref.symlink_to(src_ref)
            else:
                # Copy
                if not dst_raw.exists():
                    shutil.copy2(src_raw, dst_raw)
                if not dst_ref.exists():
                    shutil.copy2(src_ref, dst_ref)

        print(f"  {split_name}/input:     {len(list(input_dir.iterdir()))} files")
        print(f"  {split_name}/reference: {len(list(reference_dir.iterdir()))} files")

    # Save the split manifest for reproducibility
    manifest = data_dir / "split_manifest.txt"
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(f"# UIEB-890 split (seed={args.seed}, n_test={args.n_test})\n")
        f.write(f"# Train: {len(train_stems)} | Test: {len(test_stems)}\n\n")
        f.write("# TEST stems:\n")
        for stem in sorted(test_stems):
            f.write(f"test {stem}\n")
        f.write("\n# TRAIN stems:\n")
        for stem in sorted(train_stems):
            f.write(f"train {stem}\n")
    print(f"\nManifest saved to: {manifest}")
    print("Done!")


if __name__ == "__main__":
    main()
