"""Detection metrics + inference-speed benchmarking for a YOLO checkpoint.

Used both standalone (evaluate one model) and by compare_models.py (baseline
vs fine-tuned). Deliberately separate from training/train.py's built-in
validation-during-training so a checkpoint can be re-evaluated later without
retraining, and so the same function can benchmark a *baseline* pretrained
model that was never trained on this project's data at all.

Usage:

    python -m evaluation.metrics --model yolo11n.pt --data data/dataset.yaml

"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ultralytics import YOLO


def evaluate_detection_metrics(model_path: str, data_yaml: str, device: str = "cpu", split: str = "test") -> dict:
    """Run Ultralytics' validation on `split` and return P/R/mAP as a plain dict.

    IMPORTANT: results.box.p / .r / .ap50 / .ap are NOT dense arrays indexed by
    raw class id — they're dense over only the classes that actually had
    ground-truth instances, in the order given by results.box.ap_class_index.
    Indexing them by raw class id silently pulls the wrong class's numbers
    whenever a dataset's classes aren't a contiguous 0..N-1 block that exactly
    matches the model's own class ordering (e.g. evaluating an 80-class COCO
    baseline against a 5-class custom dataset.yaml — see evaluation/coco_overlap.py
    for why raw class ids don't mean the same thing across those two models).
    """
    model = YOLO(model_path)
    results = model.val(data=data_yaml, device=device, split=split, verbose=False)

    class_names = results.names
    per_class = {}
    for pos, cls_id in enumerate(results.box.ap_class_index):
        name = class_names[int(cls_id)]
        per_class[name] = {
            "precision": float(results.box.p[pos]),
            "recall": float(results.box.r[pos]),
            "ap50": float(results.box.ap50[pos]),
            "ap50_95": float(results.box.ap[pos]),
        }

    return {
        "model": model_path,
        "overall": {
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "map50": float(results.box.map50),
            "map50_95": float(results.box.map),
        },
        "per_class": per_class,
    }


def benchmark_speed(model_path: str, data_yaml: str, device: str = "cpu", imgsz: int = 416, n_images: int = 60) -> dict:
    """Time raw inference (no NMS/plotting overhead beyond predict()) over up to
    n_images from the dataset's test split, and report average latency + FPS.
    First inference is excluded from the average (warmup: model/graph init cost).
    """
    import yaml

    with open(data_yaml) as f:
        meta = yaml.safe_load(f)
    base = Path(meta["path"])
    test_dir = base / meta.get("test", meta["val"])
    image_paths = sorted(test_dir.glob("*.jpg"))[:n_images] or sorted(test_dir.glob("*.png"))[:n_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {test_dir}")

    model = YOLO(model_path)

    # Warmup (first call includes model/graph setup, not representative).
    model.predict(str(image_paths[0]), device=device, imgsz=imgsz, verbose=False)

    latencies_ms = []
    for p in image_paths:
        t0 = time.perf_counter()
        model.predict(str(p), device=device, imgsz=imgsz, verbose=False)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    avg_latency_ms = sum(latencies_ms) / len(latencies_ms)
    return {
        "model": model_path,
        "device": device,
        "imgsz": imgsz,
        "n_images": len(image_paths),
        "avg_latency_ms": avg_latency_ms,
        "fps": 1000.0 / avg_latency_ms,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a YOLO checkpoint's detection metrics + inference speed")
    p.add_argument("--model", required=True, help="Path or name of the model checkpoint (e.g. yolo11n.pt)")
    p.add_argument("--data", default="data/dataset.yaml", help="Path to dataset.yaml")
    p.add_argument("--device", default="cpu")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=416)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    metrics = evaluate_detection_metrics(args.model, args.data, args.device, args.split)
    speed = benchmark_speed(args.model, args.data, args.device, args.imgsz)

    print("=" * 50)
    print(f"Model: {args.model}  |  device: {args.device}  |  split: {args.split}")
    print("-" * 50)
    o = metrics["overall"]
    print(f"Precision: {o['precision']:.3f}  Recall: {o['recall']:.3f}  mAP50: {o['map50']:.3f}  mAP50-95: {o['map50_95']:.3f}")
    print("-" * 50)
    for name, m in metrics["per_class"].items():
        print(f"  {name:<12} P={m['precision']:.3f}  R={m['recall']:.3f}  AP50={m['ap50']:.3f}  AP50-95={m['ap50_95']:.3f}")
    print("-" * 50)
    print(f"Avg latency: {speed['avg_latency_ms']:.2f} ms  |  FPS: {speed['fps']:.2f}  (n={speed['n_images']} images)")
    print("=" * 50)
