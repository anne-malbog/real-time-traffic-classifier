"""Unit tests for the shared per-frame analytics pipeline (src/video_processor.py)."""

from src.counting import CountingLine
from src.event_detection import EventLog, StoppedVehicleDetector
from src.speed_estimation import SpeedCalibration
from src.tracking import Track
from src.video_processor import TrafficAnalyticsPipeline


def make_track(track_id, trajectory, timestamps=None, class_name="Car"):
    timestamps = timestamps or [float(i) for i in range(len(trajectory))]
    x, y = trajectory[-1]
    return Track(
        track_id=track_id, class_id=0, class_name=class_name, confidence=0.9, bbox=(x - 5, y - 5, x + 5, y + 5),
        first_seen=timestamps[0], last_seen=timestamps[-1],
        trajectory=list(trajectory), timestamps=timestamps,
    )


def make_minimal_scene(**overrides):
    scene = {
        "counting_lines": [CountingLine(name="main", p1=(0, 100), p2=(200, 100))],
        "density_thresholds": None,
        "congestion_config": None,
        "direction_reference": "screen",
        "wrong_way_detector": None,
        "stopped_vehicle_detector": StoppedVehicleDetector(movement_threshold_px=5.0, min_stopped_duration_sec=5.0),
        "restricted_zones": [],
        "speed_calibration": None,
    }
    scene.update(overrides)
    return scene


def test_process_returns_directions_for_moving_tracks():
    pipeline = TrafficAnalyticsPipeline(make_minimal_scene())
    track = make_track(1, [(0, 0), (100, 0)])  # moving RIGHT
    result = pipeline.process([track], timestamp=1.0)
    assert result.directions[1] == "RIGHT"


def test_process_updates_counter():
    pipeline = TrafficAnalyticsPipeline(make_minimal_scene())
    track = make_track(1, [(100, 50), (100, 150)])  # crosses y=100
    pipeline.process([track], timestamp=1.0)
    assert pipeline.counter.results["main"].total == 1


def test_process_computes_density_label():
    pipeline = TrafficAnalyticsPipeline(make_minimal_scene())
    tracks = [make_track(i, [(i * 10, 0), (i * 10, 5)]) for i in range(3)]
    result = pipeline.process(tracks, timestamp=1.0)
    assert result.density_label in ("LOW", "MODERATE", "HIGH", "CONGESTED")


def test_process_without_event_log_produces_no_events_or_alerts():
    pipeline = TrafficAnalyticsPipeline(make_minimal_scene())
    still = make_track(1, [(50, 50), (50, 50)])
    result = pipeline.process([still], timestamp=100.0, event_log=None)
    assert result.new_events == []
    assert result.alerts == {}


def test_process_with_event_log_flags_stopped_vehicle():
    pipeline = TrafficAnalyticsPipeline(make_minimal_scene())
    log = EventLog()
    still = make_track(1, [(50, 50), (50, 50)])
    for t in [0.0, 2.0, 4.0, 6.0]:
        result = pipeline.process([still], timestamp=t, event_log=log)
    assert len(log.events) == 1
    assert log.events[0].event_type == "stopped_vehicle"
    assert result.alerts.get(1, "").startswith("STOPPED")


def test_process_speed_disabled_by_default():
    calibration = SpeedCalibration(point1=(0, 0), point2=(100, 0), real_world_distance_m=10.0)
    pipeline = TrafficAnalyticsPipeline(make_minimal_scene(speed_calibration=calibration))
    track = make_track(1, [(0, 0), (100, 0)], timestamps=[0.0, 2.0])
    result = pipeline.process([track], timestamp=2.0, compute_speed=False)
    assert result.speed_labels == {}
    assert result.avg_speed_kmh is None


def test_process_speed_enabled_computes_labels():
    calibration = SpeedCalibration(point1=(0, 0), point2=(100, 0), real_world_distance_m=10.0)
    pipeline = TrafficAnalyticsPipeline(make_minimal_scene(speed_calibration=calibration))
    track = make_track(1, [(0, 0), (100, 0)], timestamps=[0.0, 2.0])
    result = pipeline.process([track], timestamp=2.0, compute_speed=True)
    assert "est." in result.speed_labels[1]
    assert result.avg_speed_kmh == pytest_approx(18.0)


def pytest_approx(value, rel=1e-6):
    import pytest
    return pytest.approx(value, rel=rel)


def test_process_no_speed_calibration_produces_no_labels_even_if_requested():
    pipeline = TrafficAnalyticsPipeline(make_minimal_scene(speed_calibration=None))
    track = make_track(1, [(0, 0), (100, 0)])
    result = pipeline.process([track], timestamp=1.0, compute_speed=True)
    assert result.speed_labels == {}
    assert result.avg_speed_kmh is None
