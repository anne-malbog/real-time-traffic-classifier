"""Real-time-style video inference pipeline: detection, optionally + tracking.

Phase 1 baseline: reads a video file, runs a pretrained (not yet fine-tuned)
YOLO model frame by frame, draws bounding boxes + labels + per-frame FPS, and
writes an annotated output video.

Phase 3 addition: pass --track to switch to persistent-ID multi-object
tracking (ByteTrack via Ultralytics), drawing each vehicle's track ID and a
trailing trajectory line, and reporting the total number of unique vehicles
observed.

Usage (from the project root, with the venv active):

    python -m src.inference --source data/raw/samples/highway_night.webm
    python -m src.inference --source data/raw/samples/highway_night.webm --track

"""

from __future__ import annotations

import argparse
import colorsys
import time
from pathlib import Path

import cv2

from ultralytics import YOLO

from src.counting import VehicleCounter
from src.detection import VehicleDetector, vehicle_class_ids_for_model
from src.event_detection import EventLog
from src.scene_config import load_scene_config
from src.tracking import VehicleTracker
from src.video_processor import TrafficAnalyticsPipeline

BOX_COLOR = (60, 200, 60)
TEXT_COLOR = (255, 255, 255)
LINE_COLOR = (0, 165, 255)
CONGESTION_COLOR = (0, 0, 255)
OK_COLOR = (0, 220, 0)
ZONE_COLOR = (0, 0, 220)
ALERT_COLOR = (0, 0, 255)


def id_to_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic, visually-distinct BGR color per track id (golden-ratio
    hue spacing keeps consecutive ids from landing on similar hues)."""
    hue = (track_id * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_detections(frame, detections, fps: float):
    for det in detections:
        p1 = (int(det.x1), int(det.y1))
        p2 = (int(det.x2), int(det.y2))
        cv2.rectangle(frame, p1, p2, BOX_COLOR, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (p1[0], max(p1[1] - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def draw_tracks(frame, tracks, fps: float, directions: dict[int, str] | None = None, alerts: dict[int, str] | None = None, speed_labels: dict[int, str] | None = None):
    directions = directions or {}
    alerts = alerts or {}
    speed_labels = speed_labels or {}
    for track in tracks:
        color = id_to_color(track.track_id)
        x1, y1, x2, y2 = (int(v) for v in track.bbox)
        box_color = ALERT_COLOR if track.track_id in alerts else color
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        label = f"{track.class_name} #{track.track_id} {track.confidence:.2f}"
        direction = directions.get(track.track_id)
        if direction:
            label += f"  {direction}"
        speed_label = speed_labels.get(track.track_id)
        if speed_label:
            label += f"  {speed_label}"
        cv2.putText(
            frame, label, (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
        alert_text = alerts.get(track.track_id)
        if alert_text:
            cv2.putText(
                frame, alert_text, (x1, min(y2 + 18, frame.shape[0] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, ALERT_COLOR, 2, cv2.LINE_AA,
            )
        # Trajectory trail: connect recent centers for this track.
        pts = [(int(x), int(y)) for x, y in track.trajectory]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color, 2)
        if pts:
            cv2.circle(frame, pts[-1], 3, color, -1)

    cv2.putText(
        frame, f"FPS: {fps:.1f}  |  Active tracks: {len(tracks)}", (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
    )
    return frame


def draw_analytics_hud(frame, counter: VehicleCounter, density_label: str, congestion: dict):
    """Draws counting lines + a HUD block: per-line totals, density status,
    congestion status."""
    for line in counter.lines:
        p1 = (int(line.p1[0]), int(line.p1[1]))
        p2 = (int(line.p2[0]), int(line.p2[1]))
        cv2.line(frame, p1, p2, LINE_COLOR, 2)
        cv2.putText(frame, line.name, (p1[0] + 8, p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LINE_COLOR, 2, cv2.LINE_AA)

    y = 60
    for name, result in counter.results.items():
        by_dir = ", ".join(f"{k}: {v}" for k, v in result.by_direction.items())
        text = f"{name} total: {result.total}  ({by_dir})"
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28

    density_color = CONGESTION_COLOR if density_label in ("HIGH", "CONGESTED") else OK_COLOR
    cv2.putText(frame, f"TRAFFIC DENSITY: {density_label}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, density_color, 2, cv2.LINE_AA)
    y += 30

    congestion_color = CONGESTION_COLOR if congestion["congestion_detected"] else OK_COLOR
    status = "DETECTED" if congestion["congestion_detected"] else "not detected"
    cv2.putText(
        frame, f"CONGESTION: {status}  (slow: {congestion['slow_fraction']*100:.0f}%)",
        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, congestion_color, 2, cv2.LINE_AA,
    )
    return frame, y + 32


def draw_zones(frame, zones):
    """Draws restricted-zone polygon outlines + labels."""
    for zone in zones:
        pts = [(int(x), int(y)) for x, y in zone.polygon]
        for i in range(len(pts)):
            cv2.line(frame, pts[i], pts[(i + 1) % len(pts)], ZONE_COLOR, 2)
        label_pt = pts[0]
        cv2.putText(frame, zone.zone_name, (label_pt[0], label_pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ZONE_COLOR, 2, cv2.LINE_AA)
    return frame


def draw_new_events(frame, y_start: int, new_events: list):
    """Prints newly-triggered events this frame as a scrolling-style list."""
    y = y_start
    for event in new_events:
        text = f"! {event.event_type.upper()}  id={event.track_id}  {event.class_name}"
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ALERT_COLOR, 2, cv2.LINE_AA)
        y += 26
    return frame


def run(
    source: str,
    model_path: str,
    output_path: str,
    conf: float,
    iou: float,
    device: str,
    show: bool,
    track: bool,
    tracker_config: str = "src/bytetrack_traffic.yaml",
    analytics: bool = False,
    scene_config_path: str = "configs/analytics.yaml",
    events: bool = False,
    events_output_dir: str = "outputs/events",
    speed: bool = False,
    imgsz: int = 1280,
) -> None:
    events = events or False
    speed = speed or False
    analytics = analytics or events or speed  # these all need the same scene config + tracking
    track = track or analytics  # analytics needs tracking (trajectories) to work

    # Class-vocabulary-aware, not hardcoded to COCO: the COCO pretrained
    # baseline, the stage-1 fine-tune, and the stage-2 fine-tune each have a
    # DIFFERENT id->name mapping (e.g. stage-2's id 1 is "car", COCO's id 1
    # is "bicycle") — filtering by a fixed COCO id list would silently
    # select the wrong classes (or nothing) for a non-COCO model. See
    # src/detection.py's vehicle_class_ids_for_model docstring.
    model_names = YOLO(model_path).names
    vehicle_classes = vehicle_class_ids_for_model(model_names)
    if track:
        tracker = VehicleTracker(
            model_path=model_path,
            tracker_config=tracker_config,
            conf_threshold=conf,
            iou_threshold=iou,
            device=device,
            classes=vehicle_classes,
            imgsz=imgsz,
        )
    else:
        detector = VehicleDetector(
            model_path=model_path,
            conf_threshold=conf,
            iou_threshold=iou,
            device=device,
            classes=vehicle_classes,
            imgsz=imgsz,
        )

    if analytics:
        scene = load_scene_config(scene_config_path)
        pipeline = TrafficAnalyticsPipeline(scene)

    if speed and (not analytics or scene.get("speed_calibration") is None):
        print("Warning: --speed requested but no speed_calibration in scene config; speed labels disabled.")
        speed = False

    if events:
        event_log = EventLog()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not (1.0 <= src_fps <= 120.0):
        # Some containers report bogus/implausible fps metadata (seen with
        # certain webm files); fall back to a sane default rather than
        # letting a garbage value corrupt track timestamps / speed proxies.
        print(f"Warning: implausible source FPS ({src_fps}), falling back to 30.0")
        src_fps = 30.0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (width, height)
    )

    frame_count = 0
    total_detections = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()
        if track:
            active_tracks = tracker.update(frame, timestamp=frame_count / src_fps)
            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0.0
            frame_count += 1
            total_detections += len(active_tracks)

            if analytics:
                now = frame_count / src_fps
                result = pipeline.process(active_tracks, now, event_log if events else None, compute_speed=speed)

                annotated = draw_tracks(frame, active_tracks, fps, result.directions, result.alerts, result.speed_labels)
                annotated, hud_y = draw_analytics_hud(annotated, pipeline.counter, result.density_label, result.congestion)
                if events:
                    annotated = draw_zones(annotated, pipeline.zone_detectors)
                    annotated = draw_new_events(annotated, hud_y, result.new_events)
            else:
                annotated = draw_tracks(frame, active_tracks, fps)
        else:
            detections = detector.detect(frame)
            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0.0
            frame_count += 1
            total_detections += len(detections)
            annotated = draw_detections(frame, detections, fps)

        writer.write(annotated)

        if show:
            cv2.imshow("Traffic Intelligence - Inference", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = time.time() - t_start
    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
    avg_dets = total_detections / frame_count if frame_count else 0.0
    mode_parts = [m for m, on in [("events", events), ("speed", speed), ("analytics", analytics and not events and not speed)] if on]
    mode = "+".join(mode_parts) if mode_parts else ("tracking" if track else "detection")
    print("=" * 50)
    print(f"Inference complete ({mode} mode)")
    print(f"  Source:               {source}")
    print(f"  Frames processed:     {frame_count}")
    print(f"  Total detections:     {total_detections}")
    print(f"  Avg detections/frame: {avg_dets:.2f}")
    if track:
        print(f"  Unique vehicle tracks observed: {len(tracker.tracks)}")
    if analytics:
        for name, result in pipeline.counter.results.items():
            print(f"  Counting line '{name}': total={result.total}  by_class={result.by_class}  by_direction={result.by_direction}")
    if events:
        by_type: dict[str, int] = {}
        for e in event_log.events:
            by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
        print(f"  Events logged: {len(event_log.events)}  by_type={by_type}")
        Path(events_output_dir).mkdir(parents=True, exist_ok=True)
        json_path = Path(events_output_dir) / "events.json"
        csv_path = Path(events_output_dir) / "events.csv"
        event_log.save_json(json_path)
        event_log.save_csv(csv_path)
        print(f"  Event log saved:      {json_path}, {csv_path}")
    print(f"  Elapsed time:         {elapsed:.2f}s")
    print(f"  Average FPS:          {avg_fps:.2f}")
    print(f"  Output video:         {output_path}")
    print("=" * 50)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Traffic video inference: detection, optionally + tracking")
    p.add_argument("--source", required=True, help="Path to input video file")
    p.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics model name or path (default: yolo11n.pt, pretrained on COCO)",
    )
    p.add_argument(
        "--output",
        default="outputs/videos/annotated.mp4",
        help="Path to save annotated output video",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    p.add_argument("--device", default="cpu", help="Inference device: cpu, cuda, cuda:0, etc.")
    p.add_argument("--show", action="store_true", help="Display a live preview window while processing")
    p.add_argument("--track", action="store_true", help="Enable multi-object tracking (persistent IDs + trajectories) instead of plain per-frame detection")
    p.add_argument("--tracker", default="src/bytetrack_traffic.yaml", help="Tracker config: src/bytetrack_traffic.yaml (default, traffic-tuned), or stock bytetrack.yaml / botsort.yaml")
    p.add_argument("--analytics", action="store_true", help="Enable counting/direction/density/congestion analytics (implies --track)")
    p.add_argument("--scene-config", default="configs/analytics.yaml", help="Path to scene analytics config YAML (counting lines, thresholds)")
    p.add_argument("--events", action="store_true", help="Enable wrong-way/stopped-vehicle/restricted-zone event detection + logging (implies --analytics)")
    p.add_argument("--events-output", default="outputs/events", help="Directory to save events.json/events.csv")
    p.add_argument("--speed", action="store_true", help="Show APPROXIMATE per-vehicle speed (requires speed_calibration in the scene config; see src/speed_estimation.py for accuracy caveats)")
    p.add_argument("--imgsz", type=int, default=1280, help="Inference resolution (default 1280, not Ultralytics' 640 — verified during Phase 8 that 640 misses small/distant vehicles the same weights can otherwise detect; costs FPS, see src/detection.py's VehicleDetector docstring)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        source=args.source,
        model_path=args.model,
        output_path=args.output,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        show=args.show,
        track=args.track,
        tracker_config=args.tracker,
        analytics=args.analytics,
        scene_config_path=args.scene_config,
        events=args.events,
        events_output_dir=args.events_output,
        speed=args.speed,
        imgsz=args.imgsz,
    )
