"""Multi-object tracking: persistent IDs + trajectories for detected vehicles.

Two layers, deliberately separated:

  - TrackManager: pure bookkeeping. Given one frame's worth of already-computed
    (track_id, class_id, class_name, confidence, bbox) tuples, it creates/
    updates Track objects, extends trajectories, and stamps first/last-seen
    timestamps. No YOLO or video dependency at all, so it's independently
    unit-testable (see tests/test_tracking.py) without a model or GPU/CPU
    inference cost.

  - VehicleTracker: runs actual detection+tracking (via Ultralytics' built-in
    ByteTrack/BoT-SORT integration, YOLO.track()) on a real video frame and
    feeds the results into a TrackManager.

The underlying tracker is ByteTrack by default (tracker_config="bytetrack.yaml"),
per the project's tech-stack choice — it's simple and effective for traffic
video and is what Ultralytics ships with out of the box, so this project reuses
that well-tested implementation rather than rewriting ByteTrack from scratch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from ultralytics import YOLO

Bbox = tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class Track:
    """A single tracked vehicle's current state + history."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Bbox
    first_seen: float
    last_seen: float
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def age(self) -> float:
        """Seconds (or frame-index units, if timestamps are frame indices)
        between first and last observation."""
        return self.last_seen - self.first_seen


class TrackManager:
    """Model-agnostic track bookkeeping — see module docstring."""

    def __init__(self, max_trajectory_len: int = 90):
        self.tracks: dict[int, Track] = {}
        self.max_trajectory_len = max_trajectory_len

    def update(
        self,
        detections: list[tuple[int, int, str, float, Bbox]],
        timestamp: float | None = None,
    ) -> list[Track]:
        """detections: list of (track_id, class_id, class_name, confidence, bbox)
        for the current frame. Returns the list of Track objects active this
        frame (in the same order as the input detections).
        """
        timestamp = time.time() if timestamp is None else timestamp
        active: list[Track] = []

        for track_id, class_id, class_name, confidence, bbox in detections:
            track = self.tracks.get(track_id)
            if track is None:
                track = Track(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                    first_seen=timestamp,
                    last_seen=timestamp,
                )
                self.tracks[track_id] = track
            else:
                track.class_id = class_id
                track.class_name = class_name
                track.confidence = confidence
                track.bbox = bbox
                track.last_seen = timestamp

            track.trajectory.append(track.center)
            track.timestamps.append(timestamp)
            if len(track.trajectory) > self.max_trajectory_len:
                track.trajectory.pop(0)
                track.timestamps.pop(0)

            active.append(track)

        return active

    def get_track(self, track_id: int) -> Track | None:
        return self.tracks.get(track_id)

    def reset(self) -> None:
        self.tracks.clear()


class VehicleTracker:
    """Runs YOLO detection + ByteTrack/BoT-SORT tracking on video frames and
    maintains persistent Track state via TrackManager.
    """

    def __init__(
        self,
        model_path: str | Path = "yolo11n.pt",
        tracker_config: str = "bytetrack.yaml",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        classes: list[int] | None = None,
        max_trajectory_len: int = 90,
        imgsz: int = 1280,
    ) -> None:
        self.model = YOLO(str(model_path))
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes
        self.manager = TrackManager(max_trajectory_len=max_trajectory_len)
        # See src/detection.py's VehicleDetector for why 1280, not Ultralytics'
        # default 640 — verified during Phase 8 that this alone (no retraining)
        # took missed distant-vehicle detections from 1 to 8 on the same frame.
        self.imgsz = imgsz

    def update(self, frame, timestamp: float | None = None) -> list[Track]:
        """Run tracking on a single BGR frame and return the active tracks."""
        results = self.model.track(
            frame,
            persist=True,  # keep ByteTrack's internal state across calls on this model
            tracker=self.tracker_config,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            classes=self.classes,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        raw: list[tuple[int, int, str, float, Bbox]] = []
        if results.boxes is not None and results.boxes.id is not None:
            for box in results.boxes:
                if box.id is None:
                    continue  # tracker hasn't confirmed/assigned an id yet
                track_id = int(box.id[0])
                cls_id = int(box.cls[0])
                cls_name = results.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                raw.append((track_id, cls_id, cls_name, conf, (x1, y1, x2, y2)))

        return self.manager.update(raw, timestamp)

    @property
    def tracks(self) -> dict[int, Track]:
        """All tracks ever seen (including ones no longer active this frame)."""
        return self.manager.tracks
