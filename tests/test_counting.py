"""Unit tests for virtual counting lines (src/counting.py)."""

from src.counting import CountingLine, VehicleCounter
from src.tracking import Track


def make_track(track_id, class_name, trajectory, timestamps=None):
    timestamps = timestamps or [float(i) for i in range(len(trajectory))]
    return Track(
        track_id=track_id,
        class_id=0,
        class_name=class_name,
        confidence=0.9,
        bbox=(0, 0, 10, 10),
        first_seen=timestamps[0],
        last_seen=timestamps[-1],
        trajectory=list(trajectory),
        timestamps=timestamps,
    )


def test_vehicle_counting_crossing_line_increments_total():
    line = CountingLine(name="main", p1=(0, 100), p2=(200, 100), label_a="A", label_b="B")
    counter = VehicleCounter([line])

    # Track moves straight down across y=100 (from y=50 to y=150 at x=100)
    track = make_track(1, "Car", [(100, 50), (100, 150)])
    events = counter.update([track])

    assert len(events) == 1
    assert counter.results["main"].total == 1
    assert counter.results["main"].by_class == {"Car": 1}


def test_vehicle_counting_direction_attribution():
    line = CountingLine(name="main", p1=(0, 100), p2=(200, 100), label_a="A", label_b="B")
    counter = VehicleCounter([line])

    down = make_track(1, "Car", [(100, 50), (100, 150)])  # ends below the line
    up = make_track(2, "Truck", [(100, 150), (100, 50)])  # ends above the line
    counter.update([down, up])

    result = counter.results["main"]
    assert result.total == 2
    assert set(result.by_direction.keys()) == {"A", "B"}
    assert sum(result.by_direction.values()) == 2


def test_vehicle_counting_same_id_never_double_counted():
    line = CountingLine(name="main", p1=(0, 100), p2=(200, 100))
    counter = VehicleCounter([line])

    track = make_track(1, "Car", [(100, 50), (100, 150)])
    counter.update([track])
    assert counter.results["main"].total == 1

    # Same id crosses back and forth repeatedly... still one vehicle.
    track2 = make_track(1, "Car", [(100, 150), (100, 50)])
    counter.update([track2])
    track3 = make_track(1, "Car", [(100, 50), (100, 150)])
    counter.update([track3])

    assert counter.results["main"].total == 1


def test_vehicle_counting_no_crossing_no_count():
    line = CountingLine(name="main", p1=(0, 100), p2=(200, 100))
    counter = VehicleCounter([line])

    # Moves entirely above the line... never crosses it.
    track = make_track(1, "Car", [(100, 10), (100, 30)])
    counter.update([track])

    assert counter.results["main"].total == 0


def test_vehicle_counting_multiple_lines_independent():
    line_a = CountingLine(name="line_a", p1=(0, 100), p2=(200, 100))
    line_b = CountingLine(name="line_b", p1=(0, 200), p2=(200, 200))
    counter = VehicleCounter([line_a, line_b])

    # Crosses only line_a
    track = make_track(1, "Car", [(100, 50), (100, 150)])
    counter.update([track])

    assert counter.results["line_a"].total == 1
    assert counter.results["line_b"].total == 0


def test_vehicle_counting_short_trajectory_ignored():
    line = CountingLine(name="main", p1=(0, 100), p2=(200, 100))
    counter = VehicleCounter([line])

    track = make_track(1, "Car", [(100, 150)])  # only one point, no "step" to test
    events = counter.update([track])

    assert events == []
    assert counter.results["main"].total == 0


def test_reset_clears_counts():
    line = CountingLine(name="main", p1=(0, 100), p2=(200, 100))
    counter = VehicleCounter([line])
    counter.update([make_track(1, "Car", [(100, 50), (100, 150)])])
    assert counter.results["main"].total == 1

    counter.reset()
    assert counter.results["main"].total == 0
    # Same id can be counted again after reset
    counter.update([make_track(1, "Car", [(100, 50), (100, 150)])])
    assert counter.results["main"].total == 1
