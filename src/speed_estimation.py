"""Approximate vehicle speed estimation via a configurable pixel-to-distance
calibration (spec section 16).

IMPORTANT — read before trusting a number this module produces:

  This is explicitly an ESTIMATE, not a calibrated, legally-admissible
  measurement. A monocular camera has no built-in sense of physical scale;
  everything here is downstream of one manual assumption an operator
  provides: "these two pixels are this many real-world meters apart."

  Two real limitations, not glossed over:

  1. A single scalar pixels-per-meter scale factor is used, derived from ONE
     reference distance. This ignores perspective — a vehicle near the
     camera covers far more pixels per real meter than the same vehicle far
     away. Speed estimates are most trustworthy near the calibration
     reference and increasingly wrong toward the rest of the frame,
     especially on roads with strong perspective (a receding street, not a
     purely side-on view). A proper fix needs either a full homography
     (>= 4 point correspondences mapping image points to a real-world
     ground plane) or known camera intrinsics/extrinsics — both out of
     scope for this project.
  2. The calibration reference itself is typically a nominal/assumed value
     (e.g. "standard lane width is ~3.5m"), not a surveyed on-site
     measurement, unless the operator does that work themselves.

  Never present this as police-grade or enforcement-grade speed measurement
  — see the README's Limitations section.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SpeedCalibration:
    """Derived from a reference line the operator measures/estimates in
    their own scene: two pixel points and the real-world distance (meters)
    between them (e.g. a lane's nominal width, a measured crosswalk length).
    """

    point1: tuple[float, float]
    point2: tuple[float, float]
    real_world_distance_m: float

    def __post_init__(self):
        if self.real_world_distance_m <= 0:
            raise ValueError("real_world_distance_m must be positive")
        if self.point1 == self.point2:
            raise ValueError("Calibration reference points cannot be identical")

    @property
    def pixels_per_meter(self) -> float:
        pixel_distance = math.hypot(self.point2[0] - self.point1[0], self.point2[1] - self.point1[1])
        return pixel_distance / self.real_world_distance_m


def estimate_speed_kmh(track, calibration: SpeedCalibration, window_sec: float | None = None) -> float | None:
    """Approximate speed in km/h, averaged over the track's recent
    trajectory (or just the last `window_sec` seconds of it, if given).
    Returns None when there isn't enough trajectory/elapsed time to
    estimate from — never fabricates a number from insufficient data.
    """
    pts, ts = track.trajectory, track.timestamps
    if len(pts) < 2:
        return None

    if window_sec is not None:
        cutoff = ts[-1] - window_sec
        start_idx = next((i for i, t in enumerate(ts) if t >= cutoff), 0)
        pts, ts = pts[start_idx:], ts[start_idx:]
        if len(pts) < 2:
            return None

    total_px = sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(1, len(pts)))
    total_time = ts[-1] - ts[0]
    if total_time <= 0:
        return None

    meters_per_sec = (total_px / total_time) / calibration.pixels_per_meter
    return meters_per_sec * 3.6


def format_speed_label(kmh: float | None) -> str:
    """Consistent display label — always carries the '(est.)' qualifier so
    this is never mistaken for a precise/authoritative measurement anywhere
    it's shown, per spec section 16.

    Deliberately no '~' prefix: verified during testing that OpenCV's
    Hershey font (used for the on-video overlay via cv2.putText) renders '~'
    as a flat horizontal squiggle that's visually indistinguishable from a
    '-' at small font sizes — real positive speeds were reading as if
    negative on screen. '(est.)' alone already conveys "this is
    approximate" without relying on a symbol that doesn't render reliably.
    """
    if kmh is None:
        return "speed: n/a"
    return f"{kmh:.0f} km/h (est.)"
