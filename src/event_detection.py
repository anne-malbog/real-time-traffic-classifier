"""Event detection: wrong-way driving, stopped vehicles, restricted-zone
violations — plus the unified structured event log (spec section 21) that
these and Phase 4's counting/congestion can all write into.

Pure logic operating on Track objects (src.tracking) — no video/model
dependency, independently unit-testable (tests/test_event_detection.py).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.traffic_analysis import DIRECTIONS_8WAY, compute_direction


@dataclass
class Event:
    timestamp: float
    event_type: str
    track_id: int | None
    class_name: str | None
    confidence: float | None
    location: tuple[float, float] | None
    details: dict = field(default_factory=dict)


class EventLog:
    """Accumulates Event objects and persists them as CSV/JSON."""

    def __init__(self):
        self.events: list[Event] = []

    def add(self, event: Event) -> None:
        self.events.append(event)

    def to_dicts(self) -> list[dict]:
        return [asdict(e) for e in self.events]

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dicts(), f, indent=2, default=str)

    def save_csv(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_dicts()
        fieldnames = ["timestamp", "event_type", "track_id", "class_name", "confidence", "location", "details"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def _opposite_directions(direction: str, tolerance_steps: int = 1) -> set[str]:
    """The set of 8-way labels considered 'wrong way' relative to `direction`
    — the true opposite (180deg) plus `tolerance_steps` neighbors on each
    side, so a vehicle doesn't need to be moving at an exact reverse angle
    to be flagged."""
    if direction not in DIRECTIONS_8WAY:
        return set()
    idx = DIRECTIONS_8WAY.index(direction)
    opposite_idx = (idx + 4) % 8
    return {DIRECTIONS_8WAY[(opposite_idx + d) % 8] for d in range(-tolerance_steps, tolerance_steps + 1)}


class WrongWayDetector:
    """Flags a track as wrong-way if its computed direction has been
    consistently opposite the expected direction for `min_consecutive`
    updates in a row (avoids flagging on one noisy frame). Each track is
    flagged at most once (spec section 18's example logs a single violation
    per vehicle, not one per frame it continues being wrong-way).
    """

    def __init__(self, expected_direction: str, min_consecutive: int = 5, tolerance_steps: int = 1):
        self.expected_direction = expected_direction
        self.opposite_directions = _opposite_directions(expected_direction, tolerance_steps)
        self.min_consecutive = min_consecutive
        self._consecutive_wrong: dict[int, int] = {}
        self._flagged: set[int] = set()

    def update(self, tracks, event_log: EventLog | None = None, timestamp: float | None = None) -> list[Event]:
        events = []
        for track in tracks:
            if track.track_id in self._flagged:
                continue
            direction = compute_direction(track.trajectory)
            if direction in self.opposite_directions:
                self._consecutive_wrong[track.track_id] = self._consecutive_wrong.get(track.track_id, 0) + 1
            else:
                self._consecutive_wrong[track.track_id] = 0

            if self._consecutive_wrong.get(track.track_id, 0) >= self.min_consecutive:
                self._flagged.add(track.track_id)
                event = Event(
                    timestamp=timestamp if timestamp is not None else track.last_seen,
                    event_type="wrong_way",
                    track_id=track.track_id,
                    class_name=track.class_name,
                    confidence=track.confidence,
                    location=track.center,
                    details={"direction": direction, "expected": self.expected_direction},
                )
                events.append(event)
                if event_log is not None:
                    event_log.add(event)
        return events

    def is_flagged(self, track_id: int) -> bool:
        return track_id in self._flagged


class StoppedVehicleDetector:
    """Flags a track as STOPPED once it's shown negligible frame-to-frame
    movement for at least `min_stopped_duration_sec` continuously. Requires
    a nontrivial default duration so ordinary red-light stops don't
    immediately trigger (spec section 19).
    """

    def __init__(self, movement_threshold_px: float = 10.0, min_stopped_duration_sec: float = 5.0):
        self.movement_threshold_px = movement_threshold_px
        self.min_stopped_duration_sec = min_stopped_duration_sec
        self._stopped_since: dict[int, float] = {}
        self._flagged: set[int] = set()

    def _is_currently_still(self, track) -> bool:
        pts = track.trajectory
        if len(pts) < 2:
            return False
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        return math.hypot(x1 - x0, y1 - y0) < self.movement_threshold_px

    def update(self, tracks, event_log: EventLog | None = None, timestamp: float | None = None) -> list[Event]:
        events = []
        for track in tracks:
            now = timestamp if timestamp is not None else track.last_seen
            if self._is_currently_still(track):
                self._stopped_since.setdefault(track.track_id, now)
                duration = now - self._stopped_since[track.track_id]
                if duration >= self.min_stopped_duration_sec and track.track_id not in self._flagged:
                    self._flagged.add(track.track_id)
                    event = Event(
                        timestamp=now,
                        event_type="stopped_vehicle",
                        track_id=track.track_id,
                        class_name=track.class_name,
                        confidence=track.confidence,
                        location=track.center,
                        details={"duration_sec": duration},
                    )
                    events.append(event)
                    if event_log is not None:
                        event_log.add(event)
            else:
                self._stopped_since.pop(track.track_id, None)
                self._flagged.discard(track.track_id)  # moving again -> can be re-flagged if it stops later
        return events

    def get_duration(self, track_id: int, timestamp: float) -> float | None:
        """Current continuous-stopped duration, or None if not currently
        stopped. Useful for live HUD display even before the threshold that
        triggers a logged event."""
        started = self._stopped_since.get(track_id)
        return None if started is None else timestamp - started


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon test (PNPOLY)."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


class RestrictedZoneDetector:
    """Flags a track when its center enters a user-defined polygonal zone.
    Each track is flagged at most once per zone — re-entering after leaving
    doesn't spam a second event (matches the once-per-id pattern used by
    VehicleCounter).
    """

    def __init__(self, zone_name: str, polygon: list[tuple[float, float]]):
        self.zone_name = zone_name
        self.polygon = polygon
        self._flagged: set[int] = set()

    def update(self, tracks, event_log: EventLog | None = None, timestamp: float | None = None) -> list[Event]:
        events = []
        for track in tracks:
            if track.track_id in self._flagged:
                continue
            if _point_in_polygon(track.center, self.polygon):
                self._flagged.add(track.track_id)
                event = Event(
                    timestamp=timestamp if timestamp is not None else track.last_seen,
                    event_type="restricted_zone",
                    track_id=track.track_id,
                    class_name=track.class_name,
                    confidence=track.confidence,
                    location=track.center,
                    details={"zone": self.zone_name},
                )
                events.append(event)
                if event_log is not None:
                    event_log.add(event)
        return events

    def is_flagged(self, track_id: int) -> bool:
        return track_id in self._flagged
