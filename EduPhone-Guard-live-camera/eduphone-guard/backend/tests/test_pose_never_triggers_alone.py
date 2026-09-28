"""
Estes testes existem para proteger uma garantia arquitetural específica,
pedida explicitamente no projeto: o contexto de postura (Pose) NUNCA pode
criar um evento sozinho, e a análise de pose NUNCA roda sobre um frame que
não tenha uma detecção de celular.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np

from app.schemas.detection import BoundingBox, Detection
from app.schemas.pose import PoseSignals
from app.vision.event_detector import EventDetector


def _det(camera_id: str, confidence: float) -> Detection:
    return Detection(
        detected_class="cell_phone",
        confidence=confidence,
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        timestamp=datetime.now(timezone.utc),
        frame_number=1,
        camera_id=camera_id,
    )


def _fake_frame() -> np.ndarray:
    return np.zeros((10, 10, 3), dtype="uint8")


def test_pose_analyzer_never_called_when_no_phone_detected():
    """Sem nenhuma detecção de celular no frame, a análise de pose não deve nem ser chamada."""
    fake_pose_analyzer = MagicMock()
    fake_pose_analyzer.available = True

    ed = EventDetector(
        min_confidence=0.7,
        min_detection_frames=2,
        window_seconds=3,
        cooldown_seconds=10,
        pose_analyzer=fake_pose_analyzer,
        suspicion_scorer=MagicMock(),
    )

    result = ed.process([], "cam-1", frame=_fake_frame(), now=datetime.now(timezone.utc))

    assert result is None
    fake_pose_analyzer.analyze.assert_not_called()


def test_event_without_frame_has_no_suspicion_but_still_triggers():
    """Sem `frame` (pose desativado para essa chamada), o celular ainda deve funcionar normalmente."""
    ed = EventDetector(min_confidence=0.7, min_detection_frames=1, window_seconds=3, cooldown_seconds=10)

    trigger = ed.process([_det("cam-1", 0.9)], "cam-1", frame=None, now=datetime.now(timezone.utc))

    assert trigger is not None
    assert trigger.suspicion is None  # pose não estava configurado — não quebra o fluxo do celular


def test_low_pose_signal_confidence_never_creates_event_without_phone_persistence():
    """
    Mesmo que o analyzer de pose "grite suspeita" (mocked para retornar sinais
    fortes), sem persistência de celular (min_detection_frames não atingido)
    nenhum evento é criado.
    """
    fake_pose_analyzer = MagicMock()
    fake_pose_analyzer.available = True
    fake_pose_analyzer.analyze.return_value = PoseSignals(
        hands_near_phone=True, head_down=True, hands_hidden_below_hip=True, pose_detected=True
    )

    ed = EventDetector(
        min_confidence=0.7,
        min_detection_frames=5,  # exige 5 frames — só vamos mandar 1
        window_seconds=3,
        cooldown_seconds=10,
        pose_analyzer=fake_pose_analyzer,
        suspicion_scorer=MagicMock(),
    )

    result = ed.process([_det("cam-1", 0.95)], "cam-1", frame=_fake_frame(), now=datetime.now(timezone.utc))

    assert result is None  # celular detectado, mas ainda não persistente — sem evento, mesmo com "sinais fortes" de pose


def test_suspicion_breakdown_attached_only_after_phone_persistence_confirmed():
    from app.vision.pose import SuspicionScorer

    fake_pose_analyzer = MagicMock()
    fake_pose_analyzer.available = True
    fake_pose_analyzer.analyze.return_value = PoseSignals(hands_near_phone=True, pose_detected=True)

    real_scorer = SuspicionScorer(
        weight_phone_visual=0.55,
        weight_hand_proximity=0.20,
        weight_head_down=0.10,
        weight_hands_hidden=0.10,
        weight_persistence=0.10,
        weight_writing_reduction=0.25,
    )

    ed = EventDetector(
        min_confidence=0.7,
        min_detection_frames=1,
        window_seconds=3,
        cooldown_seconds=10,
        pose_analyzer=fake_pose_analyzer,
        suspicion_scorer=real_scorer,
    )

    trigger = ed.process([_det("cam-1", 0.9)], "cam-1", frame=_fake_frame(), now=datetime.now(timezone.utc))

    assert trigger is not None
    assert trigger.suspicion is not None
    assert trigger.suspicion.phone_confidence == trigger.confidence
    assert trigger.suspicion.pose_signals.hands_near_phone is True
