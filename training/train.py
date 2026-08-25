"""Phase 2, step 3 / Phase 8: Train the YOLO model.

This script fine-tunes YOLO using the prepared traffic dataset and tracks 
each training run with MLflow. It records the training settings, performance metrics, 
and the best model so different experiments can be compared.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import yaml
from ultralytics import YOLO

from evaluation.metrics import benchmark_speed, evaluate_detection_metrics

DEFAULT_CONFIG_PATH = Path("training/config.yaml")

AUGMENTATION_DEFAULTS = {
    "scale": 0.5,          # random scale jitter gain, range [1-scale, 1+scale]
    "translate": 0.1,      # random translation, fraction of image size
    "hsv_v": 0.4,           # HSV value/brightness jitter
    "mosaic": 1.0,          # mosaic (4-image tile) augmentation
    "close_mosaic": 10,     # epochs before the end where mosaic is disabled
    "copy_paste": 0.0,      # copy-paste augmentation
    "copy_paste_mode": "flip",  # "flip" (mirror-paste)
}


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(cfg: dict) -> Path:
    ## Runs training + evaluation, logs everything to MLflow, returns the best.pt weights path.
    print("=" * 50)
    print("Training configuration")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("=" * 50)


    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(cfg.get("mlflow_experiment", "traffic-yolo-baseline"))
    with mlflow.start_run(run_name=cfg["name"]):
        aug = {k: cfg.get(k, default) for k, default in AUGMENTATION_DEFAULTS.items()}

        mlflow.log_params(
            {
                "model": cfg["model"],
                "dataset": cfg["data"],
                "epochs": cfg["epochs"],
                "batch": cfg["batch"],
                "lr0": cfg["lr0"],
                "imgsz": cfg["imgsz"],
                "device": cfg["device"],
                "conf": cfg["conf"],
                "iou": cfg["iou"],
                "patience": cfg["patience"],
                **{f"aug_{k}": v for k, v in aug.items()},
            }
        )

        model = YOLO(cfg["model"])
        results = model.train(
            data=cfg["data"],
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            lr0=cfg["lr0"],
            device=cfg["device"],
            conf=cfg["conf"],
            iou=cfg["iou"],
            patience=cfg["patience"],
            workers=cfg["workers"],

            project=str(Path(cfg["project"]).resolve()),
            name=cfg["name"],
            exist_ok=True,
            **aug,
        )

        save_dir = Path(results.save_dir)
        best_weights = save_dir / "weights" / "best.pt"

        metrics = evaluate_detection_metrics(str(best_weights), cfg["data"], device=cfg["device"], split="test")
        speed = benchmark_speed(str(best_weights), cfg["data"], device=cfg["device"], imgsz=cfg["imgsz"])

        mlflow.log_metrics(
            {
                "precision": metrics["overall"]["precision"],
                "recall": metrics["overall"]["recall"],
                "map50": metrics["overall"]["map50"],
                "map50_95": metrics["overall"]["map50_95"],
                "fps": speed["fps"],
                "avg_latency_ms": speed["avg_latency_ms"],
            }
        )
        mlflow.log_artifact(str(best_weights))

        print("=" * 50)
        print("Training complete")
        print(f"  Run directory: {save_dir}")
        print(f"  Best weights:  {best_weights}")
        print(f"  MLflow run:    {mlflow.active_run().info.run_id} (experiment: {cfg.get('mlflow_experiment', 'traffic-yolo-baseline')})")
        print(f"  Test metrics:  P={metrics['overall']['precision']:.3f} R={metrics['overall']['recall']:.3f} "
              f"mAP50={metrics['overall']['map50']:.3f} mAP50-95={metrics['overall']['map50_95']:.3f} FPS={speed['fps']:.2f}")
        print("=" * 50)

    return best_weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLO on the traffic dataset, tracked in MLflow")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to base config YAML")
    p.add_argument("--data", help="Path to dataset.yaml")
    p.add_argument("--model", help="Base checkpoint (e.g. yolo11n.pt) or path to resume from")
    p.add_argument("--epochs", type=int, help="Number of training epochs")
    p.add_argument("--imgsz", type=int, help="Training image size")
    p.add_argument("--batch", type=int, help="Batch size")
    p.add_argument("--lr0", type=float, help="Initial learning rate")
    p.add_argument("--device", help="cpu, 0, 0,1, etc.")
    p.add_argument("--conf", type=float, help="Confidence threshold for in-training validation")
    p.add_argument("--iou", type=float, help="IoU threshold for in-training validation")
    p.add_argument("--patience", type=int, help="Early-stopping patience (epochs)")
    p.add_argument("--workers", type=int, help="Dataloader workers")
    p.add_argument("--project", help="Output directory for training runs")
    p.add_argument("--name", help="Run name (subfolder under --project, and the MLflow run name)")
    p.add_argument("--mlflow-experiment", dest="mlflow_experiment", help="MLflow experiment name (e.g. traffic-yolo-baseline, traffic-yolo-augmented)")
    p.add_argument("--scale", type=float, help="Random scale jitter gain, range [1-scale, 1+scale] (default 0.5)")
    p.add_argument("--translate", type=float, help="Random translation, fraction of image size (default 0.1)")
    p.add_argument("--hsv-v", dest="hsv_v", type=float, help="HSV value/brightness jitter (default 0.4)")
    p.add_argument("--mosaic", type=float, help="Probability of mosaic augmentation (default 1.0)")
    p.add_argument("--close-mosaic", dest="close_mosaic", type=int, help="Epochs before the end where mosaic is disabled (default 10)")
    p.add_argument("--copy-paste", dest="copy_paste", type=float, help="Probability of copy-paste augmentation (default 0.0)")
    p.add_argument("--copy-paste-mode", dest="copy_paste_mode", choices=["flip", "mixup"], help="Copy-paste implementation (default 'flip')")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(Path(args.config))

    # CLI flags override config.yaml values, but only the ones actually passed.
    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    cfg.update(overrides)

    train(cfg)
