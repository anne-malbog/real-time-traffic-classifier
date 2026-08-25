"""Phase 2, step 5: dataset analysis (spec section 6.7).

Reports total images, per-split counts, object counts (overall and per
class), class imbalance, and image-resolution distribution; generates a
class-distribution chart, an images-per-split chart, a resolution chart, and
a grid of sample ground-truth annotations.

Usage:

    python -m training.analyze_dataset
    python -m training.analyze_dataset --dataset-yaml data/dataset_stage2.yaml \\
        --processed-dir data/processed_stage2 --output-dir outputs/metrics/dataset_analysis_stage2

"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

from evaluation.plots import COLOR_GRID, COLOR_SURFACE, COLOR_TEXT_MUTED, COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, style_axis

# Fixed categorical order
# assigned to classes in dataset.yaml order and never reassigned/cycled.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

SPLITS = ["train", "val", "test"]


def load_class_names(dataset_yaml_path: str = "data/dataset.yaml") -> list[str]:
    with open(dataset_yaml_path) as f:
        meta = yaml.safe_load(f)
    names = meta["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def collect_stats(class_names: list[str], processed_dir: Path) -> dict:
    stats = {
        "images_per_split": {},
        "objects_per_split_class": {s: Counter() for s in SPLITS},
        "resolutions": [],
    }
    for split in SPLITS:
        img_dir = processed_dir / "images" / split
        label_dir = processed_dir / "labels" / split
        if not img_dir.exists():
            continue
        images = sorted(img_dir.glob("*.jpg"))
        stats["images_per_split"][split] = len(images)

        for img_path in images:
            with Image_open_size(img_path) as size:
                stats["resolutions"].append(size)
            label_path = label_dir / (img_path.stem + ".txt")
            if label_path.exists():
                for line in label_path.read_text().splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    stats["objects_per_split_class"][split][class_names[cls_id]] += 1
    return stats


class Image_open_size:

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        img = cv2.imread(str(self.path))
        h, w = img.shape[:2]
        return (w, h)

    def __exit__(self, *a):
        return False


def print_summary(stats: dict, class_names: list[str]) -> None:
    total_images = sum(stats["images_per_split"].values())
    total_objects = sum(sum(c.values()) for c in stats["objects_per_split_class"].values())

    print("=" * 55)
    print("DATASET ANALYSIS")
    print("=" * 55)
    print(f"Total images: {total_images}")
    for split in SPLITS:
        print(f"  {split}: {stats['images_per_split'].get(split, 0)}")
    print(f"Total objects (all splits): {total_objects}")

    overall_class_counts = Counter()
    for c in stats["objects_per_split_class"].values():
        overall_class_counts.update(c)

    print("\nObjects per class (all splits):")
    for name in class_names:
        print(f"  {name:<12} {overall_class_counts.get(name, 0)}")

    if overall_class_counts:
        max_c, min_c = max(overall_class_counts.values()), min(overall_class_counts.values())
        imbalance_ratio = max_c / min_c if min_c > 0 else float("inf")
        print(f"\nClass imbalance ratio (max/min): {imbalance_ratio:.2f}x")

    resolutions = stats["resolutions"]
    unique_res = Counter(resolutions)
    print(f"\nImage resolutions: {len(unique_res)} unique size(s) across {len(resolutions)} images")
    for res, count in unique_res.most_common(5):
        print(f"  {res[0]}x{res[1]}: {count} images")
    print("=" * 55)


def plot_images_per_split(stats: dict, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=COLOR_SURFACE)
    style_axis(ax)
    splits = [s for s in SPLITS if s in stats["images_per_split"]]
    counts = [stats["images_per_split"][s] for s in splits]
    bars = ax.bar(splits, counts, color=CATEGORICAL[: len(splits)], zorder=3)
    for b in bars:
        ax.annotate(str(int(b.get_height())), xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", color=COLOR_TEXT_PRIMARY, fontsize=9)
    ax.set_title("Images per split", color=COLOR_TEXT_SECONDARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_class_distribution(stats: dict, class_names: list[str], save_path: Path) -> None:
    overall = Counter()
    for c in stats["objects_per_split_class"].values():
        overall.update(c)

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=COLOR_SURFACE)
    style_axis(ax)
    counts = [overall.get(name, 0) for name in class_names]
    bars = ax.bar(class_names, counts, color=CATEGORICAL[: len(class_names)], zorder=3)
    for b in bars:
        ax.annotate(str(int(b.get_height())), xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", color=COLOR_TEXT_PRIMARY, fontsize=9)
    ax.set_title("Object instances per class (all splits)", color=COLOR_TEXT_SECONDARY, fontsize=11)
    ax.tick_params(axis="x", colors=COLOR_TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_resolution_distribution(stats: dict, save_path: Path) -> None:
    resolutions = stats["resolutions"]
    unique_res = Counter(resolutions)

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=COLOR_SURFACE)
    style_axis(ax)
    labels = [f"{w}x{h}" for (w, h) in unique_res.keys()]
    counts = list(unique_res.values())
    bars = ax.bar(labels, counts, color=CATEGORICAL[0], zorder=3)
    for b in bars:
        ax.annotate(str(int(b.get_height())), xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", color=COLOR_TEXT_PRIMARY, fontsize=9)
    ax.set_title("Image resolution distribution", color=COLOR_TEXT_SECONDARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_sample_annotations(class_names: list[str], save_path: Path, processed_dir: Path, n: int = 6, seed: int = 0) -> None:
    rng = random.Random(seed)
    img_dir = processed_dir / "images" / "train"
    label_dir = processed_dir / "labels" / "train"
    candidates = [p for p in img_dir.glob("*.jpg") if (label_dir / (p.stem + ".txt")).stat().st_size > 0]
    sample = rng.sample(candidates, min(n, len(candidates)))

    cols = 3
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), facecolor=COLOR_SURFACE)
    axes = np.array(axes).reshape(-1)

    for ax, img_path in zip(axes, sample):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        label_path = label_dir / (img_path.stem + ".txt")
        for line in label_path.read_text().splitlines():
            cls_id, xc, yc, bw, bh = line.split()
            cls_id = int(cls_id)
            xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
            x1, y1 = xc - bw / 2, yc - bh / 2
            color = CATEGORICAL[cls_id % len(CATEGORICAL)]
            rect = plt.Rectangle((x1, y1), bw, bh, fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, max(y1 - 4, 0), class_names[cls_id], color=color, fontsize=9, fontweight="bold")
        ax.imshow(img)
        ax.axis("off")
    for ax in axes[len(sample):]:
        ax.axis("off")

    fig.suptitle("Sample ground-truth annotations", color=COLOR_TEXT_PRIMARY, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)
    print(f"Saved {save_path}")


def main(dataset_yaml_path: str, processed_dir: str, output_dir: str) -> None:
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)

    class_names = load_class_names(dataset_yaml_path)
    stats = collect_stats(class_names, processed_dir)
    print_summary(stats, class_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_images_per_split(stats, output_dir / "images_per_split.png")
    plot_class_distribution(stats, class_names, output_dir / "class_distribution.png")
    plot_resolution_distribution(stats, output_dir / "resolution_distribution.png")
    plot_sample_annotations(class_names, output_dir / "sample_annotations.png", processed_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset analysis: counts, imbalance, resolutions, sample annotations")
    p.add_argument("--dataset-yaml", default="data/dataset.yaml")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--output-dir", default="outputs/metrics/dataset_analysis")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.dataset_yaml, args.processed_dir, args.output_dir)
