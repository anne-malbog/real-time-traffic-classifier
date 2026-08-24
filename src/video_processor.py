"""Shared per-frame analytics pipeline: counting -> traffic analysis ->
event detection -> speed estimation.

Both src/inference.py (CLI, writes annotated video to disk) and
dashboard/app.py (Streamlit, renders live) call into this — the actual
analytics logic lives in exactly one place, not duplicated between them.
Detection + tracking themselves stay in src/tracking.py (VehicleTracker);
this module picks up from "here are this frame's active tracks."
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.counting import VehicleCounter
from src.event_detection import Event, EventLog
from src.speed_estimation import estimate_speed_kmh, format_speed_label
from src.traffic_analysis import compute_congestion, compute_density, compute_direction, compute_track_speed_px_per_sec


@dataclass
class FrameAnalytics:
    directions: dict[int, str]
    alerts: dict[int, str]
    speed_labels: dict[int, str]
    density_label: str
    congestion: dict
    new_events: list[Event]
    avg_speed_kmh: float | None


class TrafficAnalyticsPipeline:
    """Bundles the stateful analytics components (the counter + event
    detectors carry state across frames) so a caller just constructs one of
    these from a loaded scene config and calls .process() per frame."""

    def __init__(self, scene: dict):
        self.scene = scene
        self.counter = VehicleCounter(scene["counting_lines"])
        self.wrong_way_detector = scene["wrong_way_detector"]
        self.stopped_detector = scene["stopped_vehicle_detector"]
        self.zone_detectors = scene["restricted_zones"]
        self.speed_calibration = scene["speed_calibration"]

    def process(
        self,
        active_tracks: list,
        timestamp: float,
        event_log: EventLog | None = None,
        compute_speed: bool = False,
    ) -> FrameAnalytics:
        self.counter.update(active_tracks)

        directions = {t.track_id: compute_direction(t.trajectory) for t in active_tracks}
        directions = {k: v for k, v in directions.items() if v is not None}

        density_label = compute_density(len(active_tracks), self.scene["density_thresholds"])
        speeds_px = [compute_track_speed_px_per_sec(t) for t in active_tracks]
        congestion = compute_congestion(density_label, speeds_px, self.scene["congestion_config"], self.scene["density_thresholds"])

        alerts: dict[int, str] = {}
        new_events: list[Event] = []
        if event_log is not None:
            if self.wrong_way_detector is not None:
                new_events += self.wrong_way_detector.update(active_tracks, event_log, timestamp)
            new_events += self.stopped_detector.update(active_tracks, event_log, timestamp)
            for zone in self.zone_detectors:
                new_events += zone.update(active_tracks, event_log, timestamp)

            for t in active_tracks:
                if self.wrong_way_detector is not None and self.wrong_way_detector.is_flagged(t.track_id):
                    alerts[t.track_id] = "WRONG WAY"
                duration = self.stopped_detector.get_duration(t.track_id, timestamp)
                if duration is not None and duration >= 2.0:  # live progress, before the log threshold
                    alerts[t.track_id] = f"STOPPED {duration:.1f}s"
                for zone in self.zone_detectors:
                    if zone.is_flagged(t.track_id):
                        alerts[t.track_id] = f"ZONE: {zone.zone_name}"

        speed_labels: dict[int, str] = {}
        kmh_values = []
        if compute_speed and self.speed_calibration is not None:
            for t in active_tracks:
                kmh = estimate_speed_kmh(t, self.speed_calibration, window_sec=2.0)
                speed_labels[t.track_id] = format_speed_label(kmh)
                if kmh is not None:
                    kmh_values.append(kmh)
        avg_speed_kmh = sum(kmh_values) / len(kmh_values) if kmh_values else None

        return FrameAnalytics(
            directions=directions,
            alerts=alerts,
            speed_labels=speed_labels,
            density_label=density_label,
            congestion=congestion,
            new_events=new_events,
            avg_speed_kmh=avg_speed_kmh,
        )
