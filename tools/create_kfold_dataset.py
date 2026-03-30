#!/usr/bin/env python3

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import StratifiedKFold


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


@dataclass(frozen=True)
class Sample:
    source_path: Path
    class_name: str
    original_split: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create stratified k-fold train/val/test classification datasets "
            "from an existing ImageFolder dataset with train/val/test splits."
        )
    )
    parser.add_argument(
        "--input_root",
        required=True,
        type=Path,
        help="Path to the existing dataset root containing train/val/test.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help=(
            "Where to write fold_0 ... fold_{k-1}. Defaults to "
            "<input_root>_<k>fold next to the input dataset."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help=(
            "Number of folds to create. Each fold uses 1/(2*k) of the full dataset for "
            "test, 1/(2*k) for val, and the rest for train. Default: 5."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for the stratified fold split. Default: 0.",
    )
    parser.add_argument(
        "--link_mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help=(
            "How to materialize files in each fold. "
            "Use symlink to avoid duplicating images. Default: symlink."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_root if it already exists.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def collect_samples(input_root: Path):
    split_names = ["train", "val", "test"]
    classes_by_split = {}
    samples = []

    for split_name in split_names:
        split_dir = input_root / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        class_names = sorted(
            child.name for child in split_dir.iterdir() if child.is_dir()
        )
        if not class_names:
            raise ValueError(f"No class directories found in {split_dir}")
        classes_by_split[split_name] = class_names

        for class_name in class_names:
            class_dir = split_dir / class_name
            for image_path in sorted(class_dir.iterdir()):
                if is_image_file(image_path):
                    samples.append(
                        Sample(
                            source_path=image_path.resolve(),
                            class_name=class_name,
                            original_split=split_name,
                        )
                    )

    reference_classes = classes_by_split[split_names[0]]
    for split_name, class_names in classes_by_split.items():
        if class_names != reference_classes:
            raise ValueError(
                f"Classes differ across splits. {split_names[0]} has {reference_classes}, "
                f"but {split_name} has {class_names}."
            )

    if not samples:
        raise ValueError(f"No images found in {input_root}")

    return samples, reference_classes


def ensure_output_root(output_root: Path, overwrite: bool):
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_root} already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def materialize_file(src: Path, dst: Path, link_mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if link_mode == "symlink":
        os.symlink(src, dst)
    elif link_mode == "hardlink":
        os.link(src, dst)
    elif link_mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {link_mode}")


def unique_destination_name(sample: Sample, destination_dir: Path) -> str:
    candidate = sample.source_path.name
    if not (destination_dir / candidate).exists():
        return candidate

    stem = sample.source_path.stem
    suffix = sample.source_path.suffix
    prefixed = f"{sample.original_split}__{stem}{suffix}"
    if not (destination_dir / prefixed).exists():
        return prefixed

    counter = 1
    while True:
        name = f"{sample.original_split}__{stem}__{counter}{suffix}"
        if not (destination_dir / name).exists():
            return name
        counter += 1


def write_fold(
    output_root: Path,
    fold_name: str,
    split_to_samples,
    link_mode: str,
):
    fold_root = output_root / fold_name
    counts = defaultdict(dict)
    manifest = {}

    for split_name, samples in split_to_samples.items():
        manifest[split_name] = defaultdict(list)
        for sample in samples:
            class_dir = fold_root / split_name / sample.class_name
            file_name = unique_destination_name(sample, class_dir)
            destination = class_dir / file_name
            materialize_file(sample.source_path, destination, link_mode)
            manifest[split_name][sample.class_name].append(
                {
                    "source": str(sample.source_path),
                    "path": str(destination),
                    "original_split": sample.original_split,
                }
            )

        counts[split_name] = dict(Counter(sample.class_name for sample in samples))

    manifest_path = fold_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "fold": fold_name,
                "counts": counts,
                "link_mode": link_mode,
                "splits": {
                    split_name: dict(class_entries)
                    for split_name, class_entries in manifest.items()
                },
            },
            handle,
            indent=2,
        )

    return counts


def main():
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else input_root.parent / f"{input_root.name}_{args.k}fold"
    )

    samples, class_names = collect_samples(input_root)
    if args.k < 2:
        raise ValueError(f"k must be at least 2. Got {args.k}.")

    labels = [sample.class_name for sample in samples]
    indices = list(range(len(samples)))
    chunk_count = 2 * args.k

    class_counts = Counter(labels)
    for class_name, count in class_counts.items():
        if count < chunk_count:
            raise ValueError(
                f"Class '{class_name}' has only {count} images, which is fewer than 2*k={chunk_count}."
            )

    ensure_output_root(output_root, args.overwrite)

    chunk_splitter = StratifiedKFold(
        n_splits=chunk_count,
        shuffle=True,
        random_state=args.seed,
    )
    chunk_indices = [
        list(test_idx)
        for _, test_idx in chunk_splitter.split(indices, labels)
    ]

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "k": args.k,
        "chunk_count": chunk_count,
        "seed": args.seed,
        "val_fraction_per_fold": 1 / chunk_count,
        "test_fraction_per_fold": 1 / chunk_count,
        "link_mode": args.link_mode,
        "classes": class_names,
        "folds": {},
    }

    for fold_index in range(args.k):
        test_idx = chunk_indices[2 * fold_index]
        val_idx = chunk_indices[2 * fold_index + 1]
        holdout_idx = set(test_idx) | set(val_idx)

        train_samples = [samples[i] for i in indices if i not in holdout_idx]
        val_samples = [samples[i] for i in val_idx]
        test_samples = [samples[i] for i in test_idx]

        fold_name = f"fold_{fold_index}"
        counts = write_fold(
            output_root,
            fold_name,
            {
                "train": train_samples,
                "val": val_samples,
                "test": test_samples,
            },
            args.link_mode,
        )
        summary["folds"][fold_name] = counts
        print(f"{fold_name}: {counts}")

    summary_path = output_root / "fold_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nWrote {args.k} folds to {output_root}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
