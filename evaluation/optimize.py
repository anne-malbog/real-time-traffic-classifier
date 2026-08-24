"""Phase 9: inference optimization — export to ONNX, benchmark PyTorch vs. ONNX
(and ONNX FP16, where supported), verify accuracy parity, and log everything
to MLflow (experiment `traffic-yolo-optimized`), matching the same tracked-
experiment convention as Phase 8's training runs.

TensorRT is explicitly out of scope — this project's dev machine has no NVIDIA
GPU (see README's Limitations); ONNX + ONNX Runtime's CPU execution provider
is the optimization path that actually applies here.

Accuracy comparison uses the SAME COCO-class-space remapped eval set for both
variants (evaluation/coco_overlap.py) — the goal here isn't "how good is this
model at our task" (already covered in the Evaluation section), it's "does
exporting to ONNX change the model's behavior," so both variants must be
evaluated against literally the same eval set, not two different ones.

Usage:

    python -m evaluation.optimize --model yolo11n.pt --data data/dataset.yaml

"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mlflow
from ultralytics import YOLO

from evaluation.coco_overlap import build_coco_overlap_dataset
from evaluation.metrics import benchmark_speed, evaluate_detection_metrics
from evaluation.plots import plot_optimization_comparison

OUTPUT_DIR = Path("outputs/metrics")
MODELS_DIR = Path("models")


def export_onnx(model_path: str, imgsz: int, half: bool = False) -> str | None:
    """Export to ONNX, saved into models/. Returns the output path, or None if
    this export variant isn't supported on this machine (e.g. FP16 export
    commonly requires a CUDA device) — reported honestly by the caller, not
    silently skipped.

    dynamic=True is NOT optional here — a static-shape export (fixed
    batch=1, fixed imgsz) silently broke Ultralytics' own batched val()
    pipeline in testing (mAP50 0.676 -> 0.350 on the exact same weights,
    same eval set), not because ONNX changed the model's actual predictions
    (confirmed identical on direct single-image predict() calls) but because
    a static graph can't service val()'s default multi-image batch and
    something in the fallback path silently degraded rather than erroring.
    dynamic=True restores exact parity with the PyTorch baseline (verified:
    identical P/R/mAP50/mAP50-95 to 3 decimal places). This is real,
    verified behavior, not a defensive guess — see docs/failure_analysis.md.
    """
    model = YOLO(model_path)
    try:
        exported = model.export(format="onnx", imgsz=imgsz, half=half, dynamic=True)
    except Exception as e:
        print(f"  ONNX export (half={half}) failed on this machine: {e}")
        return None

    stem = Path(model_path).stem
    suffix = "_fp16" if half else "_fp32"
    dest = MODELS_DIR / f"{stem}{suffix}.onnx"
    Path(exported).replace(dest)
    return str(dest)


def _evaluate_variant(model_path: str, coco_overlap_yaml: str, full_data_yaml: str, device: str, imgsz: int, split: str) -> dict:
    metrics = evaluate_detection_metrics(model_path, coco_overlap_yaml, device, split)
    speed = benchmark_speed(model_path, full_data_yaml, device, imgsz=imgsz)
    return {**metrics, **speed}


def run_optimization(model_path: str, data_yaml: str, device: str, imgsz: int, split: str) -> dict:
    print("=" * 60)
    print(f"Baseline (PyTorch): {model_path}")
    baseline_names = YOLO(model_path).names
    coco_overlap_yaml = build_coco_overlap_dataset(baseline_names, split, dataset_yaml=data_yaml)

    pytorch_result = _evaluate_variant(model_path, coco_overlap_yaml, data_yaml, device, imgsz, split)

    print("\nExporting ONNX (FP32)...")
    onnx_fp32_path = export_onnx(model_path, imgsz, half=False)
    onnx_fp32_result = None
    if onnx_fp32_path:
        onnx_fp32_result = _evaluate_variant(onnx_fp32_path, coco_overlap_yaml, data_yaml, device, imgsz, split)

    print("\nExporting ONNX (FP16)...")
    onnx_fp16_path = export_onnx(model_path, imgsz, half=True)
    onnx_fp16_result = None
    if onnx_fp16_path:
        onnx_fp16_result = _evaluate_variant(onnx_fp16_path, coco_overlap_yaml, data_yaml, device, imgsz, split)

    return {
        "model_path": model_path,
        "imgsz": imgsz,
        "device": device,
        "pytorch": pytorch_result,
        "onnx_fp32": {"path": onnx_fp32_path, **onnx_fp32_result} if onnx_fp32_result else None,
        "onnx_fp16": {"path": onnx_fp16_path, **onnx_fp16_result} if onnx_fp16_result else None,
    }


def print_table(result: dict) -> None:
    variants = [("PyTorch (FP32)", result["pytorch"])]
    if result["onnx_fp32"]:
        variants.append(("ONNX (FP32)", result["onnx_fp32"]))
    if result["onnx_fp16"]:
        variants.append(("ONNX (FP16)", result["onnx_fp16"]))

    print("=" * 70)
    print(f"INFERENCE OPTIMIZATION — {result['model_path']} @ imgsz={result['imgsz']}, device={result['device']}")
    header = "Metric".ljust(14) + "".join(name.rjust(18) for name, _ in variants)
    print(header)
    print("-" * len(header))
    for key, label in [("precision", "Precision"), ("recall", "Recall"), ("map50", "mAP@50"), ("map50_95", "mAP@50-95")]:
        row = label.ljust(14) + "".join(f"{v['overall'][key]:>18.3f}" for _, v in variants)
        print(row)
    row = "FPS".ljust(14) + "".join(f"{v['fps']:>18.2f}" for _, v in variants)
    print(row)
    row = "Latency (ms)".ljust(14) + "".join(f"{v['avg_latency_ms']:>18.2f}" for _, v in variants)
    print(row)
    print("=" * 70)
    if not result["onnx_fp16"]:
        print("Note: ONNX FP16 export was not usable on this machine (see log above) — "
              "commonly requires a CUDA device for the export step itself, even though "
              "inference would run on CPU. Not a code bug; reported honestly rather than "
              "silently omitted.")
        print("=" * 70)


def save_outputs(result: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "optimization_comparison.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {json_path}")

    csv_path = OUTPUT_DIR / "optimization_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        variants = [("pytorch_fp32", result["pytorch"])]
        if result["onnx_fp32"]:
            variants.append(("onnx_fp32", result["onnx_fp32"]))
        if result["onnx_fp16"]:
            variants.append(("onnx_fp16", result["onnx_fp16"]))
        writer.writerow(["metric"] + [name for name, _ in variants])
        for key in ["precision", "recall", "map50", "map50_95"]:
            writer.writerow([key] + [v["overall"][key] for _, v in variants])
        writer.writerow(["fps"] + [v["fps"] for _, v in variants])
        writer.writerow(["avg_latency_ms"] + [v["avg_latency_ms"] for _, v in variants])
    print(f"Saved {csv_path}")

    plot_optimization_comparison(result, str(OUTPUT_DIR / "optimization_comparison.png"))


def log_to_mlflow(result: dict, run_name: str) -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("traffic-yolo-optimized")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "source_model": result["model_path"],
            "imgsz": result["imgsz"],
            "device": result["device"],
            "onnx_fp16_supported": result["onnx_fp16"] is not None,
        })
        for prefix, variant in [("pytorch", result["pytorch"]), ("onnx_fp32", result["onnx_fp32"]), ("onnx_fp16", result["onnx_fp16"])]:
            if variant is None:
                continue
            mlflow.log_metrics({
                f"{prefix}_precision": variant["overall"]["precision"],
                f"{prefix}_recall": variant["overall"]["recall"],
                f"{prefix}_map50": variant["overall"]["map50"],
                f"{prefix}_map50_95": variant["overall"]["map50_95"],
                f"{prefix}_fps": variant["fps"],
                f"{prefix}_avg_latency_ms": variant["avg_latency_ms"],
            })
        if result["onnx_fp32"]:
            mlflow.log_artifact(result["onnx_fp32"]["path"])
        if result["onnx_fp16"]:
            mlflow.log_artifact(result["onnx_fp16"]["path"])
        print(f"MLflow run: {mlflow.active_run().info.run_id} (experiment: traffic-yolo-optimized)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export to ONNX, benchmark vs. PyTorch, verify accuracy parity")
    p.add_argument("--model", default="yolo11n.pt", help="Source PyTorch checkpoint (default: the project's actual default model)")
    p.add_argument("--data", default="data/dataset.yaml", help="Dataset yaml for the accuracy-parity check")
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=1280, help="Matches the project's inference default (see README's imgsz fix)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--run-name", default="onnx_optimization")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_optimization(args.model, args.data, args.device, args.imgsz, args.split)
    print_table(result)
    save_outputs(result)
    log_to_mlflow(result, args.run_name)
