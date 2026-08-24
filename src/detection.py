"""YOLO-based vehicle detector.

Wraps an Ultralytics YOLO model to produce a clean, framework-agnostic
list of Detection objects, so the rest of the pipeline (tracking, counting,
analytics) never has to know about Ultralytics' internal Results API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

# COCO class ids for the vehicle classes this project targets. Shared by
# detection, tracking, and inference so "which classes count as a vehicle"
# is defined in exactly one place.
COCO_VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Vehicle-related class NAMES (lowercase) across every model vocabulary this
# project uses — COCO's own names, plus stage-1's (Ambulance/Bus/Car/
# Motorcycle/Truck) and stage-2's (bus/car/microbus/motorbike/pickup-van/
# truck). A purpose-built fine-tuned model's class ids do NOT line up with
# COCO_VEHICLE_CLASSES' ids (e.g. stage-2's id 1 is "car", but COCO's id 1 is
# "bicycle") — see evaluation/coco_overlap.py's docstring for the same
# name-vs-id gotcha on the evaluation side. vehicle_class_ids_for_model()
# below resolves this by matching on NAME against whichever model is
# actually loaded, so src/inference.py's --model flag works correctly with
# the COCO baseline, the stage-1 fine-tune, or the stage-2 fine-tune alike.
VEHICLE_CLASS_NAMES = {
    "bicycle", "car", "motorcycle", "motorbike", "bus", "truck",
    "ambulance", "microbus", "pickup-van", "van", "minibus",
}


def vehicle_class_ids_for_model(model_names: dict[int, str]) -> list[int] | None:
    """Given a loaded model's own id->name dict, return the class ids that
    match VEHICLE_CLASS_NAMES (case-insensitive) — usable directly as the
    `classes=` filter for that SAME model's .predict()/.track() calls.

    Returns None (meaning "don't filter, use every class") if nothing
    matches — e.g. a fully custom model whose class names aren't in the
    known list at all; better to pass everything through than silently
    detect nothing.
    """
    ids = [cid for cid, name in model_names.items() if name.lower() in VEHICLE_CLASS_NAMES]
    return ids or None


@dataclass(frozen=True)
class Detection:
    """A single detected object in one frame, in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class VehicleDetector:
    """Thin wrapper around an Ultralytics YOLO model for vehicle detection."""

    def __init__(
        self,
        model_path: str | Path = "yolo11n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        classes: list[int] | None = None,
        imgsz: int = 1280,
    ) -> None:
        self.model = YOLO(str(model_path))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes  # None = all classes the model knows
        # Default 1280, not Ultralytics' own default of 640. Verified during
        # Phase 8 failure investigation (golden_gate_bridge_night.webm missing
        # distant cars): on the same frame, going 640->1280 took detections
        # from 1->8 with the SAME pretrained weights — no retraining involved,
        # a smaller inference resolution was simply discarding real signal on
        # small/distant vehicles. Costs FPS (roughly 640:28fps -> 1280:11fps
        # on this CPU) — 1280 was chosen as the practical balance; 1920 gained
        # only one more detection for a further ~2.5x slowdown (~4fps, too
        # slow to call "real-time").
        self.imgsz = imgsz

    def detect(self, frame) -> list[Detection]:
        """Run detection on a single BGR frame (as read by OpenCV)."""
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            classes=self.classes,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            detections.append(Detection(x1, y1, x2, y2, conf, cls_id, cls_name))
        return detections
