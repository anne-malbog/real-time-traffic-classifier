# models/

Trained/fine-tuned model weights are **not** committed to git (see `.gitignore`) —
they're large binaries and belong in release assets / external storage instead.

This file tracks what should live here and where to get it.

| File | Stage | Source | Notes |
|---|---|---|---|
| `yolo11n.pt` | Pretrained baseline | Auto-downloaded by Ultralytics on first run (COCO weights) | Not traffic-specific; used for Phase 1 smoke tests and as the baseline in the baseline-vs-fine-tuned comparison |
| `traffic_yolo11n_finetuned.pt` | Fine-tuned (stage-1) | Produced by `training/train.py` (Phase 2) | Trained on the project's stage-1 traffic dataset — see `data/dataset.yaml` and the root README's Dataset section. (Current stage-1 checkpoint actually lives at `outputs/training_runs/stage1_finetune/weights/best.pt`; not yet copied here.) |
| `stage2_augmented.pt` | Fine-tuned (stage-2, augmented) | Produced by `notebooks/stage2_augmented_training.ipynb` on Colab (Phase 8, `training/config_stage2_augmented.yaml`), MLflow experiment `traffic-yolo-augmented` | **Not the project's default model.** Beats the COCO baseline on its own held-out test split, but generalizes poorly to this project's own sample footage (domain shift — see root README's "Stage-2 retrain with tuned augmentation" section for the full, honest before/after numbers). Kept as a documented failure case + a working example of the tracked fine-tuning pipeline, not a production recommendation. |
| `yolo11n_fp32.onnx` | Optimized (ONNX, FP32) | Produced by `evaluation/optimize.py` (Phase 9) from `yolo11n.pt`, dynamic-shape export, MLflow experiment `traffic-yolo-optimized` | **The recommended optimized artifact.** Verified bit-identical accuracy to `yolo11n.pt` (same P/R/mAP50/mAP50-95) and ~11% faster on this CPU — see root README's "Inference Optimization" section. |
| `yolo11n_fp16.onnx` | Optimized (ONNX, FP16) | Produced by `evaluation/optimize.py` (Phase 9) from `yolo11n.pt`, dynamic-shape export | **Not recommended** — accuracy-identical to the FP32 variants but measured *slower* than both on this CPU (no dedicated FP16 compute path on consumer CPUs, unlike GPU tensor cores). Kept as a documented negative result, not a production choice. |

To reproduce the pretrained baseline locally:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```
