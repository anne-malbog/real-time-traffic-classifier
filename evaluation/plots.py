"""Charts for the baseline-vs-fine-tuned comparison and training curves.

Color/layout choices follow the project's data-viz conventions: a fixed
2-slot categorical pair (never cycled), one y-axis per panel (detection
metrics 0-1 and speed FPS/ms are different scales, so they're separate
panels rather than a dual-axis chart), direct value labels since these are
static PNGs with no hover available, and muted gridlines/ink rather than
default matplotlib styling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Fixed categorical slots — slot 1 (blue) always baseline, slot 2 (orange)
# always fine-tuned. Never swapped or reused for a different meaning.
COLOR_BASELINE = "#2a78d6"
COLOR_FINETUNED = "#eb6834"
COLOR_SURFACE = "#fcfcfb"
COLOR_GRID = "#e1e0d9"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_TEXT_MUTED = "#898781"


def style_axis(ax):
    ax.set_facecolor(COLOR_SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT_MUTED, length=0)
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def _bar_labels(ax, bars, fmt="{:.2f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLOR_TEXT_PRIMARY,
        )


def plot_baseline_vs_finetuned(baseline: dict, finetuned: dict, save_path: str) -> None:
    """baseline/finetuned: dicts with 'overall' (precision/recall/map50/map50_95)
    and speed ('fps', 'avg_latency_ms') keys, as produced by evaluation.metrics
    and evaluation.compare_models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=COLOR_SURFACE)
    fig.suptitle("Baseline (pretrained) vs. Fine-tuned", color=COLOR_TEXT_PRIMARY, fontsize=13, fontweight="bold")

    # Panel 1: detection quality metrics (0-1 scale)
    metric_keys = ["precision", "recall", "map50", "map50_95"]
    metric_labels = ["Precision", "Recall", "mAP50", "mAP50-95"]
    x = np.arange(len(metric_keys))
    width = 0.35

    baseline_vals = [baseline["overall"][k] for k in metric_keys]
    finetuned_vals = [finetuned["overall"][k] for k in metric_keys]

    style_axis(ax1)
    b1 = ax1.bar(x - width / 2, baseline_vals, width, label="Baseline", color=COLOR_BASELINE, zorder=3)
    b2 = ax1.bar(x + width / 2, finetuned_vals, width, label="Fine-tuned", color=COLOR_FINETUNED, zorder=3)
    _bar_labels(ax1, b1)
    _bar_labels(ax1, b2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, color=COLOR_TEXT_SECONDARY)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Detection metrics", color=COLOR_TEXT_SECONDARY, fontsize=10)
    ax1.legend(frameon=False, labelcolor=COLOR_TEXT_SECONDARY, loc="upper right")

    # Panel 2: inference speed (FPS) — different scale, own axis, not dual-axis
    speed_x = np.arange(1)
    style_axis(ax2)
    sb1 = ax2.bar(speed_x - width / 2, [baseline["fps"]], width, color=COLOR_BASELINE, zorder=3)
    sb2 = ax2.bar(speed_x + width / 2, [finetuned["fps"]], width, color=COLOR_FINETUNED, zorder=3)
    _bar_labels(ax2, sb1, fmt="{:.1f}")
    _bar_labels(ax2, sb2, fmt="{:.1f}")
    ax2.set_xticks(speed_x)
    ax2.set_xticklabels(["FPS"], color=COLOR_TEXT_SECONDARY)
    ax2.set_title(f"Inference speed ({baseline['device']})", color=COLOR_TEXT_SECONDARY, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)
    print(f"Saved comparison chart: {save_path}")


# Fixed categorical slots for the optimization comparison — same 2-color rule
# (never cycled), reusing COLOR_BASELINE/COLOR_FINETUNED's slots since a 3rd
# variant (ONNX FP16) is optional/machine-dependent, not always present.
_VARIANT_COLORS = [COLOR_BASELINE, COLOR_FINETUNED, "#8a5fc2"]


def plot_optimization_comparison(result: dict, save_path: str) -> None:
    """result: as produced by evaluation.optimize.run_optimization — a dict
    with 'pytorch', 'onnx_fp32', 'onnx_fp16' (the latter possibly None if
    unsupported on this machine), each holding 'overall' + 'fps'/'avg_latency_ms'.
    """
    variants = [("PyTorch\n(FP32)", result["pytorch"])]
    if result["onnx_fp32"]:
        variants.append(("ONNX\n(FP32)", result["onnx_fp32"]))
    if result["onnx_fp16"]:
        variants.append(("ONNX\n(FP16)", result["onnx_fp16"]))
    colors = _VARIANT_COLORS[: len(variants)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=COLOR_SURFACE)
    fig.suptitle("Inference optimization: PyTorch vs. ONNX", color=COLOR_TEXT_PRIMARY, fontsize=13, fontweight="bold")

    # Panel 1: accuracy parity (mAP50 — the single number that matters most
    # for "did exporting change behavior")
    style_axis(ax1)
    map_vals = [v["overall"]["map50"] for _, v in variants]
    bars = ax1.bar([n for n, _ in variants], map_vals, color=colors, zorder=3, width=0.5)
    _bar_labels(ax1, bars)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("mAP@50 (accuracy parity)", color=COLOR_TEXT_SECONDARY, fontsize=10)

    # Panel 2: speed (FPS) — the actual point of this comparison
    style_axis(ax2)
    fps_vals = [v["fps"] for _, v in variants]
    bars2 = ax2.bar([n for n, _ in variants], fps_vals, color=colors, zorder=3, width=0.5)
    _bar_labels(ax2, bars2, fmt="{:.1f}")
    ax2.set_title(f"Inference speed ({result['pytorch']['device']})", color=COLOR_TEXT_SECONDARY, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)
    print(f"Saved optimization comparison chart: {save_path}")
