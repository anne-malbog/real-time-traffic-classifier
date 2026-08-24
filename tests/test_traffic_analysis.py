"""Unit tests for direction/density/congestion analysis (src/traffic_analysis.py)."""

from src.traffic_analysis import (
    CongestionConfig,
    DensityLevel,
    compute_congestion,
    compute_density,
    compute_direction,
    compute_track_speed_px_per_sec,
)
from src.tracking import Track


def make_track(trajectory, timestamps=None):
    timestamps = timestamps or [float(i) for i in range(len(trajectory))]
    return Track(
        track_id=1, class_id=0, class_name="Car", confidence=0.9, bbox=(0, 0, 10, 10),
        first_seen=timestamps[0], last_seen=timestamps[-1],
        trajectory=list(trajectory), timestamps=timestamps,
    )


# --- direction ---------------------------------------------------------

def test_direction_moving_right():
    assert compute_direction([(0, 0), (100, 0)]) == "RIGHT"


def test_direction_moving_left():
    assert compute_direction([(100, 0), (0, 0)]) == "LEFT"


def test_direction_moving_up_on_screen():
    # screen y decreases going "up"
    assert compute_direction([(0, 100), (0, 0)]) == "UP"


def test_direction_moving_down_on_screen():
    assert compute_direction([(0, 0), (0, 100)]) == "DOWN"


def test_direction_diagonal():
    assert compute_direction([(0, 100), (100, 0)]) == "UP-RIGHT"


def test_direction_none_when_barely_moving():
    assert compute_direction([(0, 0), (1, 0)], min_displacement=3.0) is None


def test_direction_none_with_single_point():
    assert compute_direction([(0, 0)]) is None


# --- density -------------------------------------------------------------

def test_density_low():
    assert compute_density(5) == "LOW"


def test_density_moderate():
    assert compute_density(15) == "MODERATE"


def test_density_high():
    assert compute_density(30) == "HIGH"


def test_density_congested():
    assert compute_density(100) == "CONGESTED"


def test_density_boundary_is_inclusive_of_lower_tier():
    assert compute_density(10) == "LOW"
    assert compute_density(11) == "MODERATE"


def test_density_custom_thresholds():
    custom = [DensityLevel("QUIET", 3), DensityLevel("BUSY", float("inf"))]
    assert compute_density(2, custom) == "QUIET"
    assert compute_density(4, custom) == "BUSY"


# --- track speed proxy ----------------------------------------------------

def test_track_speed_zero_for_stationary():
    track = make_track([(0, 0), (0, 0)], timestamps=[0.0, 1.0])
    assert compute_track_speed_px_per_sec(track) == 0.0


def test_track_speed_computed_correctly():
    # moves 30px over 2 seconds -> 15 px/sec average
    track = make_track([(0, 0), (30, 0)], timestamps=[0.0, 2.0])
    assert compute_track_speed_px_per_sec(track) == 15.0


def test_track_speed_zero_time_delta_safe():
    track = make_track([(0, 0), (30, 0)], timestamps=[1.0, 1.0])
    assert compute_track_speed_px_per_sec(track) == 0.0


# --- congestion (multi-signal) --------------------------------------------

def test_congestion_not_detected_from_density_alone():
    # HIGH density but all vehicles moving fast -> no congestion
    result = compute_congestion("HIGH", [50.0, 60.0, 55.0])
    assert result["density_sufficient"] is True
    assert result["speed_sufficient"] is False
    assert result["congestion_detected"] is False


def test_congestion_not_detected_from_slow_speed_alone():
    # Everyone slow, but density is LOW -> no congestion
    result = compute_congestion("LOW", [1.0, 2.0, 1.0])
    assert result["density_sufficient"] is False
    assert result["speed_sufficient"] is True
    assert result["congestion_detected"] is False


def test_congestion_detected_when_both_signals_agree():
    result = compute_congestion("HIGH", [1.0, 2.0, 1.0, 0.5])
    assert result["density_sufficient"] is True
    assert result["speed_sufficient"] is True
    assert result["congestion_detected"] is True


def test_congestion_custom_config_thresholds():
    config = CongestionConfig(density_trigger="MODERATE", slow_speed_threshold=5.0, min_slow_fraction=1.0)
    # MODERATE density is enough with this config; all speeds must be < 5.0
    result = compute_congestion("MODERATE", [4.0, 4.5], config=config)
    assert result["congestion_detected"] is True

    result2 = compute_congestion("MODERATE", [4.0, 6.0], config=config)  # not ALL slow
    assert result2["congestion_detected"] is False


def test_congestion_no_active_tracks_is_not_congested():
    result = compute_congestion("HIGH", [])
    assert result["slow_fraction"] == 0.0
    assert result["congestion_detected"] is False
