"""
Orquestra a decisão "devo criar um evento agora?" combinando:

1. Filtro de confiança mínima.
2. Persistência (PersistenceTracker): várias detecções dentro de uma janela.
3. Cooldown por câmera: depois de criar um evento, ignora novas detecções
   da mesma câmera por um período, evitando dezenas de eventos duplicados
   referentes ao mesmo episódio contínuo de uso do celular.
4. (Opcional) Contexto de postura (Pose): SOMENTE como reforço/atenuante de
   um evento que o celular já disparou — nunca como gatilho independente.
   Ver app.vision.pose para a justificativa dessa restrição.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np

from app.schemas.detection import Detection
from app.schemas.pose import PoseSignals, SuspicionBreakdown
from app.vision.pose import PoseAnalyzer, PoseWindowTracker, SuspicionScorer
from app.vision.tracker import PersistenceTracker

logger = logging.getLogger("eduphone_guard.vision.event_detector")


@dataclass
class EventTrigger:
    camera_id: str
    triggered_at: datetime
    confidence: float
    supporting_detections: List[Detection]
    # None quando o Pose está desabilitado ou indisponível — o evento
    # continua sendo criado normalmente, só sem o reforço/atenuante de
    # contexto corporal (o detector de celular nunca é bloqueado por isso).
    suspicion: Optional[SuspicionBreakdown] = None


class EventDetector:
    def __init__(
        self,
        min_confidence: float,
        min_detection_frames: int,
        window_seconds: float,
        cooldown_seconds: float,
        pose_analyzer: Optional[PoseAnalyzer] = None,
        suspicion_scorer: Optional[SuspicionScorer] = None,
    ) -> None:
        self._tracker = PersistenceTracker(
            window_seconds=window_seconds,
            min_detection_frames=min_detection_frames,
            min_confidence=min_confidence,
        )
        self._cooldown_seconds = cooldown_seconds
        self._cooldown_until: Dict[str, datetime] = {}
        self._recent_supporting: Dict[str, List[Detection]] = {}

        # Componentes de Pose são opcionais e independentes: se forem None
        # (ou o modelo de pose não carregar), o EventDetector se comporta
        # exatamente como antes de o Pose existir.
        self._pose_analyzer = pose_analyzer
        self._suspicion_scorer = suspicion_scorer
        self._pose_window = PoseWindowTracker(window_seconds=window_seconds) if pose_analyzer else None
        self._last_pose_signals: Dict[str, PoseSignals] = {}

    def _in_cooldown(self, camera_id: str, now: datetime) -> bool:
        until = self._cooldown_until.get(camera_id)
        return until is not None and now < until

    def _observe_pose_if_enabled(
        self, detections: List[Detection], camera_id: str, frame: Optional[np.ndarray], now: datetime
    ) -> None:
        """
        Roda a análise de pose SOMENTE quando há pelo menos uma detecção de
        celular neste frame (garantia estrutural: nunca há observação de
        pose "no vazio", sem evidência visual de celular no mesmo instante).
        """
        if self._pose_analyzer is None or not self._pose_analyzer.available:
            return
        if frame is None or not detections:
            return

        best_detection = max(detections, key=lambda d: d.confidence)
        signals = self._pose_analyzer.analyze(frame, best_detection)
        self._last_pose_signals[camera_id] = signals
        self._pose_window.observe(camera_id, signals, now=now)

    def process(
        self,
        detections: List[Detection],
        camera_id: str,
        frame: Optional[np.ndarray] = None,
        now: Optional[datetime] = None,
    ) -> Optional[EventTrigger]:
        """
        Processa as detecções de um frame para uma câmera e retorna um
        EventTrigger se um novo evento deve ser criado, ou None caso
        contrário (ainda não persistente, ou em cooldown).

        `frame` é opcional e usado apenas para a análise de pose (contexto).
        Omitir `frame` desativa o reforço de pose para esta chamada, sem
        afetar a detecção de celular em si.
        """
        now = now or datetime.now(timezone.utc)

        if self._in_cooldown(camera_id, now):
            return None

        triggered = False
        for detection in detections:
            self._recent_supporting.setdefault(camera_id, []).append(detection)
            if self._tracker.register(detection):
                triggered = True

        # A observação de pose acontece independentemente de já ter
        # disparado ou não, para que a "persistência do contexto" (quantos
        # frames recentes tiveram mãos perto do celular / cabeça baixa)
        # tenha uma janela real para medir — mas note que ela só roda
        # porque `detections` não está vazio (há celular neste frame).
        self._observe_pose_if_enabled(detections, camera_id, frame, now)

        if not triggered:
            return None

        confidence = self._tracker.best_confidence(camera_id)
        supporting = self._recent_supporting.get(camera_id, [])[-50:]  # cap para não crescer sem limite

        suspicion: Optional[SuspicionBreakdown] = None
        if self._suspicion_scorer is not None and self._pose_window is not None:
            signals = self._last_pose_signals.get(camera_id, PoseSignals(pose_detected=False))
            signals = signals.model_copy(update={"persistence_ratio": self._pose_window.persistence_ratio(camera_id)})
            suspicion = self._suspicion_scorer.score(phone_confidence=confidence, pose_signals=signals)

        # Ativa cooldown e limpa as janelas (celular e pose) para não gerar
        # eventos duplicados a partir das mesmas detecções/observações.
        self._cooldown_until[camera_id] = now + timedelta(seconds=self._cooldown_seconds)
        self._tracker.reset(camera_id)
        self._recent_supporting[camera_id] = []
        if self._pose_window is not None:
            self._pose_window.reset(camera_id)
        self._last_pose_signals.pop(camera_id, None)

        logger.info(
            "event_detected camera_id=%s confidence=%.3f suspicion_score=%s",
            camera_id,
            confidence,
            f"{suspicion.suspicion_score:.3f}" if suspicion else "n/a (pose desabilitado)",
        )

        return EventTrigger(
            camera_id=camera_id,
            triggered_at=now,
            confidence=confidence,
            supporting_detections=supporting,
            suspicion=suspicion,
        )
