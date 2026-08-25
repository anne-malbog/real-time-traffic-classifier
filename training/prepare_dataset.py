"""Phase 2, step 2: organize a raw downloaded dataset into the project layout.

Validates images, organizes them into train/val/test folders,
creates an 80/10/10 split if needed, and generates dataset.yaml.
The original dataset is kept unchanged.

"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image

# Roboflow's "valid" split name -> this project's "val" convention.
SPLIT_MAP = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _collect_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path | None]]:
    pairs = []
    for img_path in sorted(images_dir.iterdir()):
        if not is_valid_image(img_path):
            print(f"  [skip] corrupt/unreadable: {img_path.name}")
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        pairs.append((img_path, label_path if label_path.exists() else None))
    return pairs


def _copy_pairs(pairs: list[tuple[Path, Path | None]], dst_split: str, output_dir: Path) -> int:
    dst_images_dir = output_dir / "images" / dst_split
    dst_labels_dir = output_dir / "labels" / dst_split
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    for img_path, label_path in pairs:
        shutil.copy2(img_path, dst_images_dir / img_path.name)
        if label_path is not None:
            shutil.copy2(label_path, dst_labels_dir / label_path.name)
        else:
            (dst_labels_dir / (img_path.stem + ".txt")).touch()
    return len(pairs)


def prepare(
    source: str,
    output_dir: str = "data/processed",
    dataset_yaml_path: str = "data/dataset.yaml",
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> None:
    source_dir = Path(source)
    output_dir = Path(output_dir)
    dataset_yaml_path = Path(dataset_yaml_path)

    source_yaml = source_dir / "data.yaml"
    if not source_yaml.exists():
        raise FileNotFoundError(f"No data.yaml found in {source_dir}")
    with open(source_yaml) as f:
        meta = yaml.safe_load(f)
    class_names = meta["names"]

    # Split's (image, label) pairs, keyed by target split name.
    pairs_by_split: dict[str, list[tuple[Path, Path | None]]] = {}
    for src_split, dst_split in SPLIT_MAP.items():
        images_dir = source_dir / src_split / "images"
        labels_dir = source_dir / src_split / "labels"
        if not images_dir.exists():
            continue
        pairs = _collect_pairs(images_dir, labels_dir)
        pairs_by_split.setdefault(dst_split, []).extend(pairs)
        print(f"{src_split}: {len(pairs)} usable images found")

    populated_splits = [s for s in ("train", "val", "test") if pairs_by_split.get(s)]
    stats = {}

    if len(populated_splits) >= 2:
        for dst_split, pairs in pairs_by_split.items():
            stats[dst_split] = _copy_pairs(pairs, dst_split, output_dir)
            print(f"-> {dst_split}: {stats[dst_split]} images copied (source split preserved)")
    else:
        # everything dumped into "train" with val/test empty
        all_pairs = [p for pairs in pairs_by_split.values() for p in pairs]
        print(f"WARNING: source split unusable (only {populated_splits or 'no'} split(s) populated) "
              f"— pooling all {len(all_pairs)} images and re-splitting "
              f"{1 - val_fraction - test_fraction:.0%}/{val_fraction:.0%}/{test_fraction:.0%} "
              f"(seed={seed}) instead of trusting the source's split.")

        rng = random.Random(seed)
        shuffled = all_pairs[:]
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_val = int(n * val_fraction)
        n_test = int(n * test_fraction)
        split_pairs = {
            "val": shuffled[:n_val],
            "test": shuffled[n_val:n_val + n_test],
            "train": shuffled[n_val + n_test:],
        }
        for dst_split, pairs in split_pairs.items():
            stats[dst_split] = _copy_pairs(pairs, dst_split, output_dir)
            print(f"-> {dst_split}: {stats[dst_split]} images copied (re-split)")

    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(dataset_yaml_path, "w") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)

    print("=" * 50)
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Images per split: {stats}")
    print(f"Wrote {dataset_yaml_path}")
    print("=" * 50)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Organize a raw YOLO dataset into the project layout")
    p.add_argument("--source", required=True, help="Path to the raw downloaded dataset (contains data.yaml)")
    p.add_argument("--output-dir", default="data/processed", help="Destination for images/labels (default: data/processed)")
    p.add_argument("--dataset-yaml", default="data/dataset.yaml", help="Path to write the resulting dataset.yaml")
    p.add_argument("--val-fraction", type=float, default=0.1, help="Val split fraction when re-splitting is needed")
    p.add_argument("--test-fraction", type=float, default=0.1, help="Test split fraction when re-splitting is needed")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible re-splitting")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(args.source, args.output_dir, args.dataset_yaml, args.val_fraction, args.test_fraction, args.seed)
