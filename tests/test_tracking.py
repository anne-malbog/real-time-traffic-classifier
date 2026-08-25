"""Unit tests for TrackManager, pure bookkeeping, no model/video dependency."""

from src.tracking import TrackManager


def test_new_detection_creates_track():
    mgr = TrackManager()
    active = mgr.update([(1, 2, "Car", 0.9, (0, 0, 10, 10))], timestamp=0.0)

    assert len(active) == 1
    track = active[0]
    assert track.track_id == 1
    assert track.class_name == "Car"
    assert track.first_seen == 0.0
    assert track.last_seen == 0.0
    assert track.trajectory == [(5.0, 5.0)]


def test_repeated_id_extends_same_track_and_updates_last_seen_only():
    mgr = TrackManager()
    mgr.update([(1, 2, "Car", 0.9, (0, 0, 10, 10))], timestamp=0.0)
    mgr.update([(1, 2, "Car", 0.95, (10, 10, 20, 20))], timestamp=1.0)

    track = mgr.get_track(1)
    assert track is not None
    assert track.first_seen == 0.0  # unchanged
    assert track.last_seen == 1.0  # updated
    assert track.confidence == 0.95  # latest value
    assert track.bbox == (10, 10, 20, 20)
    assert track.trajectory == [(5.0, 5.0), (15.0, 15.0)]
    assert track.timestamps == [0.0, 1.0]


def test_distinct_ids_do_not_interfere():
    mgr = TrackManager()
    mgr.update(
        [
            (1, 2, "Car", 0.9, (0, 0, 10, 10)),
            (2, 5, "Bus", 0.8, (100, 100, 140, 140)),
        ],
        timestamp=0.0,
    )

    assert set(mgr.tracks.keys()) == {1, 2}
    assert mgr.get_track(1).class_name == "Car"
    assert mgr.get_track(2).class_name == "Bus"
    assert mgr.get_track(1).center == (5.0, 5.0)
    assert mgr.get_track(2).center == (120.0, 120.0)


def test_trajectory_respects_max_length_cap():
    mgr = TrackManager(max_trajectory_len=3)
    for i in range(5):
        mgr.update([(1, 2, "Car", 0.9, (i, i, i + 10, i + 10))], timestamp=float(i))

    track = mgr.get_track(1)
    assert len(track.trajectory) == 3
    assert len(track.timestamps) == 3
    # oldest two points (from i=0, i=1) should have been dropped
    assert track.timestamps == [2.0, 3.0, 4.0]
    # first_seen still reflects the true first observation, not the trimmed window
    assert track.first_seen == 0.0
    assert track.last_seen == 4.0


def test_class_can_flicker_between_updates_without_losing_identity():
    """Tracker can misclassify a frame; the track identity (id) should 
    persist even if class_name changes between updates."""
    mgr = TrackManager()
    mgr.update([(1, 2, "Car", 0.9, (0, 0, 10, 10))], timestamp=0.0)
    mgr.update([(1, 7, "Truck", 0.6, (1, 1, 11, 11))], timestamp=1.0)

    track = mgr.get_track(1)
    assert track.track_id == 1
    assert track.class_name == "Truck"  # latest wins
    assert len(track.trajectory) == 2  # history preserved regardless


def test_reset_clears_all_tracks():
    mgr = TrackManager()
    mgr.update([(1, 2, "Car", 0.9, (0, 0, 10, 10))], timestamp=0.0)
    assert len(mgr.tracks) == 1

    mgr.reset()
    assert len(mgr.tracks) == 0
    assert mgr.get_track(1) is None


def test_empty_frame_returns_no_active_tracks_but_preserves_existing():
    mgr = TrackManager()
    mgr.update([(1, 2, "Car", 0.9, (0, 0, 10, 10))], timestamp=0.0)
    active = mgr.update([], timestamp=1.0)

    assert active == []
    # the track from frame 0 still exists in history, just wasn't active this frame
    assert mgr.get_track(1) is not None
    assert mgr.get_track(1).last_seen == 0.0  # not updated since it wasn't seen
