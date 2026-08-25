"""Phase 7: Streamlit analytics dashboard.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Running as `streamlit run dashboard/app.py` from the project root without project installed as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.plots import COLOR_GRID, COLOR_SURFACE, COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, style_axis
from ultralytics import YOLO

from src.detection import vehicle_class_ids_for_model
from src.event_detection import EventLog
from src.inference import draw_analytics_hud, draw_new_events, draw_tracks, draw_zones
from src.scene_config import load_scene_config
from src.traffic_analysis import DEFAULT_DENSITY_THRESHOLDS
from src.tracking import VehicleTracker
from src.video_processor import TrafficAnalyticsPipeline

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SAMPLE_DIR = Path("data/raw/samples")

st.set_page_config(page_title="Real-Time Traffic Intelligence", layout="wide")
st.title("Real-Time Traffic Intelligence")
st.caption(
    "Live detection + tracking + counting + direction/density/congestion + wrong-way/"
    "stopped/restricted-zone events + approximate speed. All running the same "
    "pipeline as `src/inference.py`, rendered live instead of written to a file."
)

# Sidebar controls

with st.sidebar:
    st.header("Configuration")

    sample_videos = sorted(SAMPLE_DIR.glob("*.webm")) if SAMPLE_DIR.exists() else []
    if not sample_videos:
        st.error(f"No sample videos found in {SAMPLE_DIR}/. See the README's Setup section.")
        st.stop()
    source_path = st.selectbox("Video source", sample_videos, format_func=lambda p: p.name)

    model_path = st.text_input("Model", value="yolo11n.pt")
    scene_config_path = st.text_input("Scene config", value="configs/analytics.yaml")
    device = st.selectbox("Device", ["cpu", "0"], help="'0' = first GPU, if available")

    st.divider()
    enable_speed = st.checkbox("Estimate speed (approximate needs calibration in scene config)", value=True)
    max_frames = st.slider("Max frames to process (demo limit)", 50, 3000, 400, step=50)
    update_every = st.slider("Refresh UI every N frames", 1, 30, 8, help="Higher = smoother but choppier updates; lower = more responsive but slower overall (CPU)")

    start = st.button("Start Analysis", type="primary", use_container_width=True)

if not start:
    st.info("Configure options in the sidebar and click **Start Analysis** to begin.")
    st.caption(
        "Note: this runs real inference on CPU (no GPU on the dev machine), expect "
        "roughly 10-30 FPS of underlying processing depending on scene density, slower "
        "still with the Streamlit UI refresh overhead."
    )
    st.stop()

# Load pipeline

scene = load_scene_config(scene_config_path)
vehicle_classes = vehicle_class_ids_for_model(YOLO(model_path).names)

with st.spinner("Loading model..."):
    tracker = VehicleTracker(
        model_path=model_path,
        tracker_config="src/bytetrack_traffic.yaml",
        device=device,
        classes=vehicle_classes,
    )
    analytics_pipeline = TrafficAnalyticsPipeline(scene)
    event_log = EventLog()

speed_enabled = enable_speed and scene["speed_calibration"] is not None
if enable_speed and not speed_enabled:
    st.warning("Speed estimation requested but no `speed_calibration` in the scene config. Disabled.")

cap = cv2.VideoCapture(str(source_path))
if not cap.isOpened():
    st.error(f"Could not open {source_path}")
    st.stop()
src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
if not (1.0 <= src_fps <= 120.0):
    src_fps = 30.0  # some webm files misreport fps metadata (verified during Phase 4)

# Layout 

video_col, metrics_col = st.columns([2, 1])
with video_col:
    st.subheader("Live Annotated Video")
    video_placeholder = st.empty()
with metrics_col:
    st.subheader("Metrics")
    metrics_placeholder = st.empty()

st.subheader("Vehicle Distribution")
dist_placeholder = st.empty()

st.subheader("Traffic Over Time")
chart_placeholder = st.empty()

st.subheader("Event Log")
events_placeholder = st.empty()

progress_bar = st.progress(0.0, text="Starting...")

# Processing loop

DENSITY_LABELS = [lvl.label for lvl in (scene["density_thresholds"] or DEFAULT_DENSITY_THRESHOLDS)]
history = {"t_min": deque(maxlen=200), "vehicles_per_min": deque(maxlen=200), "density_rank": deque(maxlen=200), "avg_speed": deque(maxlen=200)}
fps_window = deque(maxlen=30)

frame_count = 0
t_start = time.time()

while frame_count < max_frames:
    ok, frame = cap.read()
    if not ok:
        break

    t0 = time.time()
    now = frame_count / src_fps
    active_tracks = tracker.update(frame, timestamp=now)
    result = analytics_pipeline.process(active_tracks, now, event_log, compute_speed=speed_enabled)
    dt = time.time() - t0
    fps_window.append(1.0 / dt if dt > 0 else 0.0)
    frame_count += 1

    annotated = draw_tracks(frame, active_tracks, fps_window[-1], result.directions, result.alerts, result.speed_labels)
    annotated, hud_y = draw_analytics_hud(annotated, analytics_pipeline.counter, result.density_label, result.congestion)
    annotated = draw_zones(annotated, analytics_pipeline.zone_detectors)
    annotated = draw_new_events(annotated, hud_y, result.new_events)

    if frame_count % update_every == 0 or frame_count == 1:
        video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        progress_bar.progress(min(frame_count / max_frames, 1.0), text=f"Frame {frame_count}/{max_frames}")

        MIN_ELAPSED_SEC_FOR_RATE = 5.0
        if now >= MIN_ELAPSED_SEC_FOR_RATE:
            vehicles_per_min = len(tracker.tracks) / (now / 60.0)
        else:
            vehicles_per_min = None
        avg_fps = sum(fps_window) / len(fps_window)

        with metrics_placeholder.container():
            c1, c2 = st.columns(2)
            c1.metric("Vehicles Detected", len(tracker.tracks))
            c2.metric("Vehicles / Minute", f"{vehicles_per_min:.1f}" if vehicles_per_min is not None else "warming up...")
            c1.metric("Traffic Density", result.density_label)
            c2.metric("Congestion", "DETECTED" if result.congestion["congestion_detected"] else "Clear")
            c1.metric("Avg Est. Speed", f"{result.avg_speed_kmh:.0f} km/h" if result.avg_speed_kmh is not None else "n/a")
            c2.metric("Current FPS", f"{avg_fps:.1f}")

        # Vehicle distribution
        by_class: dict[str, int] = {}
        for line_result in analytics_pipeline.counter.results.values():
            for k, v in line_result.by_class.items():
                by_class[k] = by_class.get(k, 0) + v
        fig1, ax1 = plt.subplots(figsize=(6, 3), facecolor=COLOR_SURFACE)
        style_axis(ax1)
        if by_class:
            names, counts = list(by_class.keys()), list(by_class.values())
            ax1.bar(names, counts, color=CATEGORICAL[: len(names)], zorder=3)
        else:
            ax1.text(0.5, 0.5, "No vehicles counted yet", ha="center", va="center", color=COLOR_TEXT_SECONDARY, transform=ax1.transAxes)
        ax1.set_title("Counted vehicles by class", color=COLOR_TEXT_SECONDARY, fontsize=10)
        fig1.tight_layout()
        dist_placeholder.pyplot(fig1)
        plt.close(fig1)

        # Traffic over time
        history["t_min"].append(now / 60.0)
        history["vehicles_per_min"].append(vehicles_per_min if vehicles_per_min is not None else float("nan"))
        history["density_rank"].append(DENSITY_LABELS.index(result.density_label) if result.density_label in DENSITY_LABELS else 0)
        history["avg_speed"].append(result.avg_speed_kmh or 0)

        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 3), facecolor=COLOR_SURFACE)
        panels = [
            ("Vehicles / minute", history["vehicles_per_min"], CATEGORICAL[0]),
            ("Traffic density", history["density_rank"], CATEGORICAL[1]),
            ("Avg. estimated speed (km/h)", history["avg_speed"], CATEGORICAL[2]),
        ]
        for ax, (title, series, color) in zip(axes2, panels):
            style_axis(ax)
            ax.plot(list(history["t_min"]), list(series), color=color, linewidth=2)
            ax.set_title(title, color=COLOR_TEXT_SECONDARY, fontsize=10)
            ax.set_xlabel("minutes", color=COLOR_TEXT_SECONDARY, fontsize=8)
            if title == "Traffic density":
                ax.set_yticks(range(len(DENSITY_LABELS)))
                ax.set_yticklabels(DENSITY_LABELS, fontsize=7)
        fig2.tight_layout()
        chart_placeholder.pyplot(fig2)
        plt.close(fig2)

        # Events table
        if event_log.events:
            df = pd.DataFrame(
                [
                    {"Timestamp": f"{e.timestamp:.1f}s", "Event": e.event_type, "Vehicle ID": e.track_id, "Class": e.class_name}
                    for e in reversed(event_log.events)
                ]
            )
            events_placeholder.dataframe(df, use_container_width=True, hide_index=True, height=220)
        else:
            events_placeholder.info("No events logged yet.")

cap.release()
progress_bar.progress(1.0, text="Done")
st.success(f"Analysis complete: processed {frame_count} frames in {time.time() - t_start:.1f}s.")
