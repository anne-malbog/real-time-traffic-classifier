"""Direction, traffic density, and (multi-signal) congestion analysis.

All pure functions operating on Track objects / plain numbers — no video or
model dependency, independently unit-testable (tests/test_traffic_analysis.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# 8-way SCREEN-relative direction labels, ordered starting at "RIGHT" (0deg)
# going counter-clockwise. Deliberately screen-relative, not compass
# (N/E/S/W): a monocular, uncalibrated camera has no way to know true
# geographic orientation — claiming "NORTH" without a calibrated compass
# reference would be fabricating precision this system doesn't have. If a
# deployment's camera orientation is known, relabel these at the call site
# (e.g. "UP" -> "NORTH") rather than this module pretending to know it.
DIRECTIONS_8WAY = ["RIGHT", "UP-RIGHT", "UP", "UP-LEFT", "LEFT", "DOWN-LEFT", "DOWN", "DOWN-RIGHT"]


def compute_direction(trajectory: list[tuple[float, float]], min_displacement: float = 3.0) -> str | None:
    """Classify overall movement direction from a trajectory window's net
    displacement (first point -> last point) into one of 8 screen-relative
    labels. Returns None when displacement is too small to be a reliable
    signal (e.g. a stationary/idling vehicle) — a direction computed from
    near-zero motion is noise, not signal.
    """
    if len(trajectory) < 2:
        return None
    x1, y1 = trajectory[0]
    x2, y2 = trajectory[-1]
    dx, dy = x2 - x1, y2 - y1
    if math.hypot(dx, dy) < min_displacement:
        return None
    # Screen y grows downward, so negate dy to get a conventional
    # counter-clockwise-from-right angle.
    angle = math.degrees(math.atan2(-dy, dx)) % 360
    return DIRECTIONS_8WAY[round(angle / 45) % 8]


@dataclass
class DensityLevel:
    label: str
    max_count: float  # inclusive upper bound; float("inf") for the top tier


DEFAULT_DENSITY_THRESHOLDS: list[DensityLevel] = [
    DensityLevel("LOW", 10),
    DensityLevel("MODERATE", 25),
    DensityLevel("HIGH", 40),
    DensityLevel("CONGESTED", float("inf")),
]


def compute_density(active_vehicle_count: int, thresholds: list[DensityLevel] | None = None) -> str:
    """Classify the current active-vehicle count into a density label using
    configurable thresholds (spec section 14 — "thresholds should be
    configurable rather than hard-coded")."""
    thresholds = thresholds or DEFAULT_DENSITY_THRESHOLDS
    for level in thresholds:
        if active_vehicle_count <= level.max_count:
            return level.label
    return thresholds[-1].label


def compute_track_speed_px_per_sec(track) -> float:
    """Average pixel-space speed over a Track's stored trajectory window.

    NOT a calibrated real-world speed (that's Phase 6, which needs an
    explicit pixel-to-distance calibration) — this is a relative "how much
    is this vehicle actually moving" proxy, used here only to help decide
    whether traffic looks congested, not to report a speed to the user.
    """
    pts, ts = track.trajectory, track.timestamps
    if len(pts) < 2:
        return 0.0
    total_dist = sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(1, len(pts)))
    total_time = ts[-1] - ts[0]
    return total_dist / total_time if total_time > 0 else 0.0


@dataclass
class CongestionConfig:
    density_trigger: str = "HIGH"  # minimum density label that can contribute to congestion
    slow_speed_threshold: float = 15.0  # px/sec proxy (see compute_track_speed_px_per_sec)
    min_slow_fraction: float = 0.5  # fraction of active tracks that must be "slow"


def compute_congestion(
    density_label: str,
    track_speeds_px_per_sec: list[float],
    config: CongestionConfig | None = None,
    density_thresholds: list[DensityLevel] | None = None,
) -> dict:
    """Multi-signal congestion detection (spec section 15 — "should not
    classify congestion from vehicle count alone"): requires BOTH a
    sufficiently high density level AND a meaningful fraction of active
    vehicles moving slowly.
    """
    config = config or CongestionConfig()
    density_thresholds = density_thresholds or DEFAULT_DENSITY_THRESHOLDS
    labels_in_order = [t.label for t in density_thresholds]

    density_rank = labels_in_order.index(density_label) if density_label in labels_in_order else 0
    trigger_rank = labels_in_order.index(config.density_trigger) if config.density_trigger in labels_in_order else 0
    density_sufficient = density_rank >= trigger_rank

    if track_speeds_px_per_sec:
        slow_fraction = sum(1 for s in track_speeds_px_per_sec if s < config.slow_speed_threshold) / len(track_speeds_px_per_sec)
    else:
        slow_fraction = 0.0
    speed_sufficient = slow_fraction >= config.min_slow_fraction

    return {
        "congestion_detected": density_sufficient and speed_sufficient,
        "density_label": density_label,
        "density_sufficient": density_sufficient,
        "slow_fraction": slow_fraction,
        "speed_sufficient": speed_sufficient,
    }
