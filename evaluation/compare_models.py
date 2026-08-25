"""Phase 2, Step 4: Compare the pretrained and fine-tuned models.

This script compares the pretrained YOLO model with the fine-tuned model. 
It evaluates only the classes both models can recognize for a fair comparison, 
while classes unique to the fine-tuned model are reported separately.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO

import yaml

from evaluation.coco_overlap import build_class_mapping, build_coco_overlap_dataset
from evaluation.metrics import benchmark_speed, evaluate_detection_metrics
from evaluation.plots import plot_baseline_vs_finetuned

OUTPUT_DIR = Path("outputs/metrics")


def _macro_average(per_class: dict, keys: list[str]) -> dict:
    metrics = ["precision", "recall", "ap50", "ap50_95"]
    out = {m: sum(per_class[k][m] for k in keys) / len(keys) for m in metrics}
    return {"precision": out["precision"], "recall": out["recall"], "map50": out["ap50"], "map50_95": out["ap50_95"]}


def run_comparison(baseline_path: str, finetuned_path: str, data_yaml: str, device: str, split: str) -> dict:
    with open(data_yaml) as f:
        our_names = yaml.safe_load(f)["names"]

    all_class_names = list(our_names.values()) if isinstance(our_names, dict) else list(our_names)

    baseline_model_names = YOLO(baseline_path).names
    class_map = build_class_mapping(all_class_names, baseline_model_names)
    overlap_classes = sorted(class_map.keys())
    dropped_classes = sorted(set(all_class_names) - set(class_map.keys()))
    print(f"Building COCO-class-space overlap eval set (classes: {overlap_classes}"
          + (f" — {', '.join(dropped_classes)} excluded, no COCO equivalent)" if dropped_classes else ")"))
    coco_overlap_yaml = build_coco_overlap_dataset(baseline_model_names, split, dataset_yaml=data_yaml)

    print(f"Evaluating baseline on overlap classes {overlap_classes}: {baseline_path}")
    baseline_metrics = evaluate_detection_metrics(baseline_path, coco_overlap_yaml, device, split)
    baseline_speed = benchmark_speed(baseline_path, data_yaml, device)
    baseline = {**baseline_metrics, **baseline_speed}

    print(f"Evaluating fine-tuned (full {len(all_class_names)} classes): {finetuned_path}")
    finetuned_metrics = evaluate_detection_metrics(finetuned_path, data_yaml, device, split)
    finetuned_speed = benchmark_speed(finetuned_path, data_yaml, device)
    finetuned_full = {**finetuned_metrics, **finetuned_speed}


    finetuned_overlap_overall = _macro_average(finetuned_metrics["per_class"], overlap_classes)
    finetuned_overlap = {
        "model": finetuned_path,
        "overall": finetuned_overlap_overall,
        "per_class": {k: finetuned_metrics["per_class"][k] for k in overlap_classes},
        "fps": finetuned_speed["fps"],
        "avg_latency_ms": finetuned_speed["avg_latency_ms"],
        "device": finetuned_speed["device"],
    }

    return {
        "baseline": baseline,
        "finetuned_overlap": finetuned_overlap,
        "finetuned_full": finetuned_full,
        "overlap_classes": overlap_classes,
        "dropped_classes": dropped_classes,
    }


def print_table(result: dict) -> None:
    b, f = result["baseline"]["overall"], result["finetuned_overlap"]["overall"]
    bs, fs = result["baseline"], result["finetuned_overlap"]
    overlap_classes = result["overlap_classes"]
    dropped_classes = result["dropped_classes"]
    print("=" * 70)
    print(f"FAIR COMPARISON — overlap classes only ({', '.join(overlap_classes)})")
    print(f"{'Metric':<14}{'Baseline':>15}{'Fine-Tuned':>15}")
    print("-" * 70)
    print(f"{'Precision':<14}{b['precision']:>15.3f}{f['precision']:>15.3f}")
    print(f"{'Recall':<14}{b['recall']:>15.3f}{f['recall']:>15.3f}")
    print(f"{'mAP@50':<14}{b['map50']:>15.3f}{f['map50']:>15.3f}")
    print(f"{'mAP@50-95':<14}{b['map50_95']:>15.3f}{f['map50_95']:>15.3f}")
    print(f"{'FPS':<14}{bs['fps']:>15.2f}{fs['fps']:>15.2f}")
    print(f"{'Latency (ms)':<14}{bs['avg_latency_ms']:>15.2f}{fs['avg_latency_ms']:>15.2f}")
    print("=" * 70)
    ff = result["finetuned_full"]["overall"]
    dropped_note = f" (includes {', '.join(dropped_classes)} — no baseline equivalent to compare against)" if dropped_classes else ""
    print(f"Fine-tuned FULL class performance{dropped_note}:")
    print(f"  Precision={ff['precision']:.3f}  Recall={ff['recall']:.3f}  mAP50={ff['map50']:.3f}  mAP50-95={ff['map50_95']:.3f}")
    print("=" * 70)


def save_outputs(result: dict, prefix: str = "baseline_vs_finetuned") -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / f"{prefix}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {json_path}")

    csv_path = OUTPUT_DIR / f"{prefix}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "baseline_overlap", "finetuned_overlap", "finetuned_full_5class"])
        b, fo, ff = result["baseline"]["overall"], result["finetuned_overlap"]["overall"], result["finetuned_full"]["overall"]
        for key, label in [("precision", "precision"), ("recall", "recall"), ("map50", "map50"), ("map50_95", "map50_95")]:
            writer.writerow([label, b[key], fo[key], ff[key]])
        writer.writerow(["fps", result["baseline"]["fps"], result["finetuned_overlap"]["fps"], result["finetuned_full"]["fps"]])
        writer.writerow(["avg_latency_ms", result["baseline"]["avg_latency_ms"], result["finetuned_overlap"]["avg_latency_ms"], result["finetuned_full"]["avg_latency_ms"]])
    print(f"Saved {csv_path}")

    plot_baseline_vs_finetuned(result["baseline"], result["finetuned_overlap"], str(OUTPUT_DIR / f"{prefix}.png"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare baseline vs. fine-tuned model (class-vocabulary-aware)")
    p.add_argument("--baseline", required=True, help="Baseline (pretrained) checkpoint")
    p.add_argument("--finetuned", required=True, help="Fine-tuned checkpoint")
    p.add_argument("--data", default="data/dataset.yaml")
    p.add_argument("--device", default="cpu")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--output-prefix",
        default="baseline_vs_finetuned",
        help="Filename prefix for outputs/metrics/<prefix>.{json,csv,png} — "
             "IMPORTANT: pick a distinct prefix per comparison (e.g. "
             "baseline_vs_stage2_augmented) or you will silently overwrite a "
             "previous run's saved comparison (default matches the stage-1 "
             "artifacts already referenced in the README).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_comparison(args.baseline, args.finetuned, args.data, args.device, args.split)
    print_table(result)
    save_outputs(result, args.output_prefix)
