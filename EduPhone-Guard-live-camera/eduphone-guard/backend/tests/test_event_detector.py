from datetime import datetime, timedelta, timezone

from app.schemas.detection import BoundingBox, Detection
from app.vision.event_detector import EventDetector


def _det(camera_id: str, confidence: float, ts: datetime, frame: int) -> Detection:
    return Detection(
        detected_class="cell_phone",
        confidence=confidence,
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        timestamp=ts,
        frame_number=frame,
        camera_id=camera_id,
    )


def test_no_event_without_persistence():
    ed = EventDetector(min_confidence=0.7, min_detection_frames=3, window_seconds=3, cooldown_seconds=10)
    now = datetime.now(timezone.utc)
    result = ed.process([_det("cam-1", 0.9, now, 1)], "cam-1", now=now)
    assert result is None


def test_event_created_after_persistent_detections():
    ed = EventDetector(min_confidence=0.7, min_detection_frames=2, window_seconds=3, cooldown_seconds=10)
    now = datetime.now(timezone.utc)
    assert ed.process([_det("cam-1", 0.9, now, 1)], "cam-1", now=now) is None
    trigger = ed.process([_det("cam-1", 0.9, now + timedelta(seconds=0.2), 2)], "cam-1", now=now + timedelta(seconds=0.2))
    assert trigger is not None
    assert trigger.camera_id == "cam-1"
    assert trigger.confidence >= 0.9


def test_cooldown_prevents_duplicate_events():
    ed = EventDetector(min_confidence=0.7, min_detection_frames=2, window_seconds=3, cooldown_seconds=30)
    now = datetime.now(timezone.utc)
    ed.process([_det("cam-1", 0.9, now, 1)], "cam-1", now=now)
    first_trigger = ed.process([_det("cam-1", 0.9, now + timedelta(seconds=0.2), 2)], "cam-1", now=now + timedelta(seconds=0.2))
    assert first_trigger is not None

    # Novas detecções logo depois, ainda dentro do cooldown, não devem gerar novo evento.
    second = ed.process(
        [_det("cam-1", 0.95, now + timedelta(seconds=1), 3)],
        "cam-1",
        now=now + timedelta(seconds=1),
    )
    assert second is None


def test_event_possible_again_after_cooldown_expires():
    ed = EventDetector(min_confidence=0.7, min_detection_frames=1, window_seconds=3, cooldown_seconds=5)
    now = datetime.now(timezone.utc)
    first = ed.process([_det("cam-1", 0.9, now, 1)], "cam-1", now=now)
    assert first is not None

    after_cooldown = now + timedelta(seconds=6)
    second = ed.process([_det("cam-1", 0.9, after_cooldown, 2)], "cam-1", now=after_cooldown)
    assert second is not None
