"""Loads a scene's spatial/analytics configuration from YAML.

Centralizing this here means later phases (Phase 5 restricted zones, Phase 6
speed calibration) extend one file's schema instead of scattering ad-hoc
yaml.safe_load() calls through the codebase — one place per deployed camera
to configure counting lines, thresholds, etc.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.counting import CountingLine
from src.event_detection import RestrictedZoneDetector, StoppedVehicleDetector, WrongWayDetector
from src.speed_estimation import SpeedCalibration
from src.traffic_analysis import CongestionConfig, DensityLevel


def load_scene_config(path: str | Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    lines = [
        CountingLine(
            name=item["name"],
            p1=tuple(item["point1"]),
            p2=tuple(item["point2"]),
            label_a=item.get("direction_a_label", "A"),
            label_b=item.get("direction_b_label", "B"),
        )
        for item in raw.get("counting_lines", [])
    ]

    density_raw = raw.get("density", {}).get("thresholds", [])
    density_thresholds = [
        DensityLevel(label=item["label"], max_count=item["max"] if item["max"] is not None else float("inf"))
        for item in density_raw
    ] or None  # None -> caller falls back to DEFAULT_DENSITY_THRESHOLDS

    congestion_raw = raw.get("congestion", {})
    congestion_config = CongestionConfig(
        density_trigger=congestion_raw.get("density_trigger", "HIGH"),
        slow_speed_threshold=congestion_raw.get("slow_speed_px_per_sec", 15.0),
        min_slow_fraction=congestion_raw.get("min_slow_fraction", 0.5),
    )

    wrong_way_detector = None
    wrong_way_raw = raw.get("wrong_way")
    if wrong_way_raw and wrong_way_raw.get("expected_direction"):
        wrong_way_detector = WrongWayDetector(
            expected_direction=wrong_way_raw["expected_direction"],
            min_consecutive=wrong_way_raw.get("min_consecutive_frames", 5),
            tolerance_steps=wrong_way_raw.get("tolerance_steps", 1),
        )

    stopped_raw = raw.get("stopped_vehicle", {})
    stopped_detector = StoppedVehicleDetector(
        movement_threshold_px=stopped_raw.get("movement_threshold_px", 10.0),
        min_stopped_duration_sec=stopped_raw.get("min_stopped_duration_sec", 5.0),
    )

    restricted_zones = [
        RestrictedZoneDetector(zone_name=item["name"], polygon=[tuple(p) for p in item["polygon"]])
        for item in raw.get("restricted_zones", [])
    ]

    speed_calibration = None
    speed_raw = raw.get("speed_calibration")
    if speed_raw:
        speed_calibration = SpeedCalibration(
            point1=tuple(speed_raw["point1"]),
            point2=tuple(speed_raw["point2"]),
            real_world_distance_m=speed_raw["real_world_distance_m"],
        )

    return {
        "counting_lines": lines,
        "density_thresholds": density_thresholds,
        "congestion_config": congestion_config,
        "direction_reference": raw.get("direction", {}).get("reference", "screen"),
        "wrong_way_detector": wrong_way_detector,
        "stopped_vehicle_detector": stopped_detector,
        "restricted_zones": restricted_zones,
        "speed_calibration": speed_calibration,
    }
