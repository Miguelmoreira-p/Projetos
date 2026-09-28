from datetime import datetime, timedelta, timezone

from app.schemas.pose import PoseSignals
from app.vision.pose import PoseWindowTracker


def test_empty_window_has_zero_ratio():
    tracker = PoseWindowTracker(window_seconds=3)
    assert tracker.persistence_ratio("cam-1") == 0.0


def test_ratio_reflects_fraction_of_flagged_frames():
    tracker = PoseWindowTracker(window_seconds=3)
    now = datetime.now(timezone.utc)
    tracker.observe("cam-1", PoseSignals(hands_near_phone=True), now=now)
    tracker.observe("cam-1", PoseSignals(hands_near_phone=False), now=now + timedelta(seconds=0.1))
    assert tracker.persistence_ratio("cam-1") == 0.5


def test_old_observations_fall_outside_window():
    tracker = PoseWindowTracker(window_seconds=1)
    now = datetime.now(timezone.utc)
    tracker.observe("cam-1", PoseSignals(hands_near_phone=True), now=now)
    tracker.observe("cam-1", PoseSignals(hands_near_phone=False), now=now + timedelta(seconds=2))
    # A primeira observação (True) já saiu da janela de 1s.
    assert tracker.persistence_ratio("cam-1") == 0.0


def test_reset_clears_window():
    tracker = PoseWindowTracker(window_seconds=3)
    now = datetime.now(timezone.utc)
    tracker.observe("cam-1", PoseSignals(hands_near_phone=True), now=now)
    tracker.reset("cam-1")
    assert tracker.persistence_ratio("cam-1") == 0.0


def test_head_down_also_counts_as_flagged():
    tracker = PoseWindowTracker(window_seconds=3)
    now = datetime.now(timezone.utc)
    tracker.observe("cam-1", PoseSignals(head_down=True), now=now)
    assert tracker.persistence_ratio("cam-1") == 1.0


def test_cameras_are_independent():
    tracker = PoseWindowTracker(window_seconds=3)
    now = datetime.now(timezone.utc)
    tracker.observe("cam-1", PoseSignals(hands_near_phone=True), now=now)
    assert tracker.persistence_ratio("cam-2") == 0.0
