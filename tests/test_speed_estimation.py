"""Unit tests for approximate speed estimation (src/speed_estimation.py)."""

import pytest

from src.speed_estimation import SpeedCalibration, estimate_speed_kmh, format_speed_label
from src.tracking import Track


def make_track(trajectory, timestamps=None):
    timestamps = timestamps or [float(i) for i in range(len(trajectory))]
    x, y = trajectory[-1]
    return Track(
        track_id=1, class_id=0, class_name="Car", confidence=0.9, bbox=(x - 5, y - 5, x + 5, y + 5),
        first_seen=timestamps[0], last_seen=timestamps[-1],
        trajectory=list(trajectory), timestamps=timestamps,
    )


# --- SpeedCalibration --------------------------------------------------------

def test_calibration_pixels_per_meter():
    # 100px spans 10m -> 10 px/m
    cal = SpeedCalibration(point1=(0, 0), point2=(100, 0), real_world_distance_m=10.0)
    assert cal.pixels_per_meter == pytest.approx(10.0)


def test_calibration_diagonal_distance():
    # 3-4-5 triangle: (0,0) to (3,4) is 5px
    cal = SpeedCalibration(point1=(0, 0), point2=(3, 4), real_world_distance_m=5.0)
    assert cal.pixels_per_meter == pytest.approx(1.0)


def test_calibration_rejects_identical_points():
    with pytest.raises(ValueError):
        SpeedCalibration(point1=(10, 10), point2=(10, 10), real_world_distance_m=5.0)


def test_calibration_rejects_nonpositive_distance():
    with pytest.raises(ValueError):
        SpeedCalibration(point1=(0, 0), point2=(100, 0), real_world_distance_m=0.0)
    with pytest.raises(ValueError):
        SpeedCalibration(point1=(0, 0), point2=(100, 0), real_world_distance_m=-5.0)


# --- estimate_speed_kmh -------------------------------------------------------

def test_estimate_speed_known_value():
    # Calibration: 10 px/m. Track moves 100px in 2s -> 50px/s -> 5 m/s -> 18 km/h
    cal = SpeedCalibration(point1=(0, 0), point2=(100, 0), real_world_distance_m=10.0)
    track = make_track([(0, 0), (100, 0)], timestamps=[0.0, 2.0])
    assert estimate_speed_kmh(track, cal) == pytest.approx(18.0)


def test_estimate_speed_stationary_is_zero():
    cal = SpeedCalibration(point1=(0, 0), point2=(100, 0), real_world_distance_m=10.0)
    track = make_track([(50, 50), (50, 50)], timestamps=[0.0, 2.0])
    assert estimate_speed_kmh(track, cal) == pytest.approx(0.0)


def test_estimate_speed_none_with_single_point():
    cal = SpeedCalibration(point1=(0, 0), point2=(100, 0), real_world_distance_m=10.0)
    track = make_track([(50, 50)])
    assert estimate_speed_kmh(track, cal) is None


def test_estimate_speed_none_with_zero_elapsed_time():
    cal = SpeedCalibration(point1=(0, 0), point2=(100, 0), real_world_distance_m=10.0)
    track = make_track([(0, 0), (100, 0)], timestamps=[1.0, 1.0])
    assert estimate_speed_kmh(track, cal) is None


def test_estimate_speed_respects_window_sec():
    # Long trajectory: fast at the start, stationary at the end.
    # A window covering only the stationary tail should report ~0.
    cal = SpeedCalibration(point1=(0, 0), point2=(100, 0), real_world_distance_m=10.0)
    track = make_track(
        [(0, 0), (500, 0), (500, 0), (500, 0)],
        timestamps=[0.0, 1.0, 2.0, 3.0],
    )
    fast = estimate_speed_kmh(track, cal)  # whole window: includes the fast jump
    windowed = estimate_speed_kmh(track, cal, window_sec=1.0)  # last ~1s only: stationary
    assert fast > 0
    assert windowed == pytest.approx(0.0)


def test_estimate_speed_none_when_window_too_short_for_two_points():
    cal = SpeedCalibration(point1=(0, 0), point2=(100, 0), real_world_distance_m=10.0)
    track = make_track([(0, 0), (100, 0)], timestamps=[0.0, 10.0])
    # A tiny window that only captures the very last point
    assert estimate_speed_kmh(track, cal, window_sec=0.001) is None


# --- format_speed_label --------------------------------------------------------

def test_format_speed_label_includes_estimate_qualifier():
    label = format_speed_label(42.3)
    assert "est." in label
    assert "42" in label


def test_format_speed_label_handles_none():
    assert format_speed_label(None) == "speed: n/a"
