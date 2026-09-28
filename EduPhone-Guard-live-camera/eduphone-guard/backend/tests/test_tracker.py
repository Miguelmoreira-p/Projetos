from datetime import datetime, timedelta, timezone

import pytest

from app.schemas.detection import BoundingBox, Detection
from app.vision.tracker import PersistenceTracker


def _make_detection(camera_id: str, confidence: float, ts: datetime, frame: int) -> Detection:
    return Detection(
        detected_class="cell_phone",
        confidence=confidence,
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        timestamp=ts,
        frame_number=frame,
        camera_id=camera_id,
    )


def test_single_detection_does_not_trigger():
    tracker = PersistenceTracker(window_seconds=3, min_detection_frames=8, min_confidence=0.7)
    now = datetime.now(timezone.utc)
    triggered = tracker.register(_make_detection("cam-1", 0.9, now, 1))
    assert triggered is False


def test_low_confidence_never_counts():
    tracker = PersistenceTracker(window_seconds=3, min_detection_frames=2, min_confidence=0.7)
    now = datetime.now(timezone.utc)
    assert tracker.register(_make_detection("cam-1", 0.5, now, 1)) is False
    assert tracker.register(_make_detection("cam-1", 0.5, now, 2)) is False


def test_persistence_triggers_after_min_frames():
    tracker = PersistenceTracker(window_seconds=3, min_detection_frames=3, min_confidence=0.7)
    now = datetime.now(timezone.utc)
    assert tracker.register(_make_detection("cam-1", 0.9, now, 1)) is False
    assert tracker.register(_make_detection("cam-1", 0.9, now + timedelta(seconds=0.5), 2)) is False
    assert tracker.register(_make_detection("cam-1", 0.9, now + timedelta(seconds=1), 3)) is True


def test_old_detections_fall_outside_window():
    tracker = PersistenceTracker(window_seconds=1, min_detection_frames=2, min_confidence=0.7)
    now = datetime.now(timezone.utc)
    assert tracker.register(_make_detection("cam-1", 0.9, now, 1)) is False
    # Segunda detecção 2 segundos depois: a primeira já saiu da janela de 1s.
    assert tracker.register(_make_detection("cam-1", 0.9, now + timedelta(seconds=2), 2)) is False


def test_reset_clears_window():
    tracker = PersistenceTracker(window_seconds=3, min_detection_frames=2, min_confidence=0.7)
    now = datetime.now(timezone.utc)
    assert tracker.register(_make_detection("cam-1", 0.9, now, 1)) is False
    tracker.reset("cam-1")
    assert tracker.register(_make_detection("cam-1", 0.9, now + timedelta(seconds=0.1), 2)) is False


def test_cameras_are_independent():
    tracker = PersistenceTracker(window_seconds=3, min_detection_frames=2, min_confidence=0.7)
    now = datetime.now(timezone.utc)
    assert tracker.register(_make_detection("cam-1", 0.9, now, 1)) is False
    assert tracker.register(_make_detection("cam-2", 0.9, now, 1)) is False
