"""Unit tests for wrong-way / stopped-vehicle / restricted-zone detection
and the event log (src/event_detection.py)."""

from src.event_detection import EventLog, RestrictedZoneDetector, StoppedVehicleDetector, WrongWayDetector
from src.tracking import Track


def make_track(track_id, trajectory, timestamps=None, class_name="Car"):
    timestamps = timestamps or [float(i) for i in range(len(trajectory))]
    # Track.center is derived from bbox, not trajectory — build a small bbox
    # centered on the trajectory's last point so track.center matches it,
    # which is what RestrictedZoneDetector actually checks.
    x, y = trajectory[-1]
    bbox = (x - 5, y - 5, x + 5, y + 5)
    return Track(
        track_id=track_id, class_id=0, class_name=class_name, confidence=0.9, bbox=bbox,
        first_seen=timestamps[0], last_seen=timestamps[-1],
        trajectory=list(trajectory), timestamps=timestamps,
    )


# --- EventLog --------------------------------------------------------------

def test_event_log_saves_json_and_csv(tmp_path):
    from src.event_detection import Event

    log = EventLog()
    log.add(Event(timestamp=1.0, event_type="stopped_vehicle", track_id=1, class_name="Car",
                   confidence=0.9, location=(10.0, 20.0), details={"duration_sec": 8.2}))

    json_path = tmp_path / "events.json"
    csv_path = tmp_path / "events.csv"
    log.save_json(json_path)
    log.save_csv(csv_path)

    assert json_path.exists()
    assert csv_path.exists()
    assert "stopped_vehicle" in json_path.read_text()
    assert "stopped_vehicle" in csv_path.read_text()


def test_event_log_empty_still_writes_header(tmp_path):
    log = EventLog()
    csv_path = tmp_path / "events.csv"
    log.save_csv(csv_path)
    content = csv_path.read_text()
    assert "event_type" in content
    assert content.count("\n") <= 2  # just header (+ possibly trailing newline)


# --- WrongWayDetector --------------------------------------------------------

def test_wrong_way_not_flagged_moving_expected_direction():
    detector = WrongWayDetector(expected_direction="DOWN", min_consecutive=3)
    log = EventLog()
    for i in range(5):
        track = make_track(1, [(0, i * 20), (0, (i + 1) * 20)])  # moving DOWN, as expected
        detector.update([track], event_log=log, timestamp=float(i))
    assert not detector.is_flagged(1)
    assert log.events == []


def test_wrong_way_flagged_after_min_consecutive_opposite():
    detector = WrongWayDetector(expected_direction="DOWN", min_consecutive=3)
    log = EventLog()
    # moving UP (opposite of DOWN) for several consecutive updates
    for i in range(5):
        track = make_track(1, [(0, 100 - i * 20), (0, 100 - (i + 1) * 20)])
        detector.update([track], event_log=log, timestamp=float(i))

    assert detector.is_flagged(1)
    assert len(log.events) == 1
    assert log.events[0].event_type == "wrong_way"
    assert log.events[0].track_id == 1


def test_wrong_way_not_flagged_below_min_consecutive():
    detector = WrongWayDetector(expected_direction="DOWN", min_consecutive=10)
    log = EventLog()
    for i in range(3):  # fewer than min_consecutive
        track = make_track(1, [(0, 100 - i * 20), (0, 100 - (i + 1) * 20)])
        detector.update([track], event_log=log, timestamp=float(i))
    assert not detector.is_flagged(1)


def test_wrong_way_resets_streak_when_direction_corrects():
    detector = WrongWayDetector(expected_direction="DOWN", min_consecutive=3)
    log = EventLog()
    wrong = make_track(1, [(0, 100), (0, 80)])   # UP (wrong)
    detector.update([wrong], event_log=log, timestamp=0.0)
    detector.update([wrong], event_log=log, timestamp=1.0)
    right = make_track(1, [(0, 80), (0, 100)])   # DOWN (correct) — resets streak
    detector.update([right], event_log=log, timestamp=2.0)
    detector.update([wrong], event_log=log, timestamp=3.0)
    detector.update([wrong], event_log=log, timestamp=4.0)

    # Only 2 consecutive wrong-way observations since the reset — not flagged yet
    assert not detector.is_flagged(1)


def test_wrong_way_flagged_only_once():
    detector = WrongWayDetector(expected_direction="DOWN", min_consecutive=2)
    log = EventLog()
    for i in range(6):
        track = make_track(1, [(0, 100 - i * 20), (0, 100 - (i + 1) * 20)])
        detector.update([track], event_log=log, timestamp=float(i))
    assert len(log.events) == 1  # not one event per subsequent frame


# --- StoppedVehicleDetector --------------------------------------------------

def test_stopped_vehicle_not_flagged_before_duration_threshold():
    detector = StoppedVehicleDetector(movement_threshold_px=5.0, min_stopped_duration_sec=5.0)
    log = EventLog()
    for t in [0.0, 1.0, 2.0]:  # only 2 seconds of stillness so far
        track = make_track(1, [(50, 50), (50, 50)], timestamps=[t, t])
        detector.update([track], event_log=log, timestamp=t)
    assert log.events == []


def test_stopped_vehicle_flagged_after_duration_threshold():
    detector = StoppedVehicleDetector(movement_threshold_px=5.0, min_stopped_duration_sec=5.0)
    log = EventLog()
    for t in [0.0, 2.0, 4.0, 6.0]:
        track = make_track(1, [(50, 50), (50, 50)], timestamps=[t, t])
        detector.update([track], event_log=log, timestamp=t)

    assert len(log.events) == 1
    assert log.events[0].event_type == "stopped_vehicle"
    assert log.events[0].details["duration_sec"] >= 5.0


def test_stopped_vehicle_not_flagged_if_moving():
    detector = StoppedVehicleDetector(movement_threshold_px=5.0, min_stopped_duration_sec=5.0)
    log = EventLog()
    for t in range(10):
        track = make_track(1, [(t * 20, 0), (t * 20 + 20, 0)], timestamps=[float(t), float(t)])
        detector.update([track], event_log=log, timestamp=float(t))
    assert log.events == []


def test_stopped_vehicle_duration_resets_after_moving_again():
    detector = StoppedVehicleDetector(movement_threshold_px=5.0, min_stopped_duration_sec=5.0)
    log = EventLog()
    still = make_track(1, [(50, 50), (50, 50)])
    detector.update([still], event_log=log, timestamp=0.0)
    detector.update([still], event_log=log, timestamp=3.0)

    moving = make_track(1, [(50, 50), (150, 50)])
    detector.update([moving], event_log=log, timestamp=4.0)  # resets the stopped clock

    detector.update([still], event_log=log, timestamp=5.0)
    detector.update([still], event_log=log, timestamp=7.0)

    # Only ~3s of continuous stillness since the reset — should not be flagged yet
    assert log.events == []


def test_get_duration_reports_live_progress():
    detector = StoppedVehicleDetector(movement_threshold_px=5.0, min_stopped_duration_sec=100.0)
    still = make_track(1, [(50, 50), (50, 50)])
    detector.update([still], timestamp=0.0)
    detector.update([still], timestamp=3.0)
    assert detector.get_duration(1, timestamp=3.0) == 3.0


def test_get_duration_none_when_not_stopped():
    detector = StoppedVehicleDetector()
    assert detector.get_duration(999, timestamp=0.0) is None


# --- RestrictedZoneDetector ---------------------------------------------------

SQUARE_ZONE = [(100, 100), (200, 100), (200, 200), (100, 200)]


def test_restricted_zone_violation_when_entering():
    detector = RestrictedZoneDetector("test_zone", SQUARE_ZONE)
    log = EventLog()
    track = make_track(1, [(150, 150)])  # center inside the square
    events = detector.update([track], event_log=log, timestamp=1.0)

    assert len(events) == 1
    assert events[0].event_type == "restricted_zone"
    assert events[0].details["zone"] == "test_zone"
    assert log.events[0] is events[0]


def test_restricted_zone_no_violation_outside():
    detector = RestrictedZoneDetector("test_zone", SQUARE_ZONE)
    track = make_track(1, [(500, 500)])  # well outside
    events = detector.update([track])
    assert events == []


def test_restricted_zone_flagged_only_once():
    detector = RestrictedZoneDetector("test_zone", SQUARE_ZONE)
    log = EventLog()
    inside_track = make_track(1, [(150, 150)])
    detector.update([inside_track], event_log=log, timestamp=1.0)
    detector.update([inside_track], event_log=log, timestamp=2.0)
    detector.update([inside_track], event_log=log, timestamp=3.0)

    assert len(log.events) == 1
    assert detector.is_flagged(1)


def test_restricted_zone_independent_tracks():
    detector = RestrictedZoneDetector("test_zone", SQUARE_ZONE)
    log = EventLog()
    inside = make_track(1, [(150, 150)])
    outside = make_track(2, [(500, 500)])
    detector.update([inside, outside], event_log=log, timestamp=1.0)

    assert detector.is_flagged(1)
    assert not detector.is_flagged(2)
