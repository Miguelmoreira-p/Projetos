"""
Análise de postura corporal (YOLO Pose) como CONTEXTO complementar ao
detector de celular — nunca como gatilho independente.

REGRA ARQUITETURAL INVIOLÁVEL: este módulo só deve ser chamado depois que
o detector de celular (app.vision.detector) já encontrou uma detecção com
confiança >= min_confidence em um frame. Nenhuma função aqui cria eventos
ou eleva suspeita a partir do nada — todas recebem uma detecção de celular
como entrada obrigatória.

O QUE NÃO SE FAZ AQUI:
- Reconhecimento facial.
- Identificação de pessoas (nome, matrícula, rosto).
- Armazenamento de keypoints brutos (coordenadas do esqueleto) além do
  escopo de um único frame em memória.

Keypoints seguem a convenção COCO-17 usada pelos modelos de pose do
Ultralytics YOLO:
  0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
  5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
  9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
  13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from app.schemas.detection import Detection
from app.schemas.pose import PoseSignals, SuspicionBreakdown

logger = logging.getLogger("eduphone_guard.vision.pose")

_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

Point = Tuple[float, float]


@dataclass
class _Keypoint:
    x: float
    y: float
    confidence: float


@dataclass
class _PersonKeypoints:
    bbox: Tuple[float, float, float, float]
    keypoints: Dict[str, _Keypoint]


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _bbox_center(bbox: Tuple[float, float, float, float]) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class PoseAnalyzer:
    """
    Extrai keypoints de pessoas em um frame usando um modelo YOLO Pose e
    deriva sinais de contexto corporal relativos a UMA detecção de celular
    específica (a mais próxima espacialmente da pessoa).
    """

    def __init__(
        self,
        model_path: str,
        min_keypoint_confidence: float = 0.5,
        hand_proximity_ratio: float = 0.6,
        head_down_ratio: float = 0.12,
        writing_hands_together_ratio: float = 0.5,
    ) -> None:
        self._min_kp_conf = min_keypoint_confidence
        self._hand_proximity_ratio = hand_proximity_ratio
        self._head_down_ratio = head_down_ratio
        self._writing_ratio = writing_hands_together_ratio
        self._model = None
        self._model_path = model_path

        try:
            from ultralytics import YOLO

            if not Path(model_path).exists() and not model_path.endswith(".pt"):
                raise FileNotFoundError(model_path)
            self._model = YOLO(model_path)
        except Exception:
            logger.warning(
                "pose_model_unavailable path=%s — análise de pose será ignorada "
                "(o detector de celular continua funcionando normalmente).",
                model_path,
            )

    @property
    def available(self) -> bool:
        return self._model is not None

    def _extract_people(self, frame: np.ndarray) -> List[_PersonKeypoints]:
        if self._model is None:
            return []

        results = self._model.predict(frame, verbose=False)
        people: List[_PersonKeypoints] = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            keypoints_result = getattr(result, "keypoints", None)
            if boxes is None or keypoints_result is None:
                continue

            xy = keypoints_result.xy  # (n_people, 17, 2)
            conf = keypoints_result.conf  # (n_people, 17)
            if xy is None or conf is None:
                continue

            for person_idx in range(xy.shape[0]):
                bbox = tuple(float(v) for v in boxes[person_idx].xyxy[0].tolist())
                kp_map: Dict[str, _Keypoint] = {}
                for kp_idx, kp_name in enumerate(_KEYPOINT_NAMES):
                    x, y = float(xy[person_idx, kp_idx, 0]), float(xy[person_idx, kp_idx, 1])
                    c = float(conf[person_idx, kp_idx])
                    kp_map[kp_name] = _Keypoint(x=x, y=y, confidence=c)

                people.append(_PersonKeypoints(bbox=bbox, keypoints=kp_map))

        return people

    def _nearest_person(
        self, people: List[_PersonKeypoints], phone_bbox_center: Point
    ) -> Optional[_PersonKeypoints]:
        if not people:
            return None
        return min(people, key=lambda p: _distance(_bbox_center(p.bbox), phone_bbox_center))

    def _valid(self, kp: Optional[_Keypoint]) -> bool:
        return kp is not None and kp.confidence >= self._min_kp_conf

    def analyze(self, frame: np.ndarray, phone_detection: Detection) -> PoseSignals:
        """
        Roda a análise de pose para UM frame, no contexto de UMA detecção de
        celular já existente. Retorna sinais derivados; nunca afirma a
        presença de celular por conta própria.
        """
        if not self.available:
            return PoseSignals(pose_detected=False)

        phone_center = _bbox_center(phone_detection.bbox.as_tuple())
        people = self._extract_people(frame)
        person = self._nearest_person(people, phone_center)

        if person is None:
            return PoseSignals(pose_detected=False)

        kp = person.keypoints

        left_shoulder, right_shoulder = kp.get("left_shoulder"), kp.get("right_shoulder")
        left_hip, right_hip = kp.get("left_hip"), kp.get("right_hip")
        left_wrist, right_wrist = kp.get("left_wrist"), kp.get("right_wrist")
        nose = kp.get("nose")

        shoulder_points = [k for k in (left_shoulder, right_shoulder) if self._valid(k)]
        hip_points = [k for k in (left_hip, right_hip) if self._valid(k)]

        if not shoulder_points:
            # Sem ombros confiáveis, não há base geométrica suficiente para
            # nenhum sinal — melhor não afirmar nada do que adivinhar.
            return PoseSignals(pose_detected=True)

        shoulder_mid = (
            sum(k.x for k in shoulder_points) / len(shoulder_points),
            sum(k.y for k in shoulder_points) / len(shoulder_points),
        )
        shoulder_width = (
            _distance((left_shoulder.x, left_shoulder.y), (right_shoulder.x, right_shoulder.y))
            if self._valid(left_shoulder) and self._valid(right_shoulder)
            else max(1.0, _distance(shoulder_mid, phone_center) * 0.3)  # estimativa grosseira de fallback
        )
        shoulder_width = max(shoulder_width, 1.0)  # evita divisão por zero

        hip_mid = (
            (sum(k.x for k in hip_points) / len(hip_points), sum(k.y for k in hip_points) / len(hip_points))
            if hip_points
            else None
        )

        wrists = [w for w in (left_wrist, right_wrist) if self._valid(w)]

        # --- hands_near_phone ---
        hands_near_phone = False
        if wrists:
            proximity_threshold = shoulder_width * self._hand_proximity_ratio
            hands_near_phone = any(
                _distance((w.x, w.y), phone_center) <= proximity_threshold for w in wrists
            )

        # --- head_down ---
        head_down = False
        if self._valid(nose) and hip_mid is not None:
            torso_height = max(_distance(shoulder_mid, hip_mid), 1.0)
            vertical_drop = nose.y - shoulder_mid[1]  # y cresce para baixo na imagem
            head_down = vertical_drop > torso_height * self._head_down_ratio

        # --- hands_hidden_below_hip ---
        hands_hidden_below_hip = False
        if len(wrists) == 2 and hip_mid is not None:
            hands_hidden_below_hip = all(w.y > hip_mid[1] for w in wrists)

        # --- writing_or_reading_pose (reduz suspeita) ---
        writing_or_reading_pose = False
        if len(wrists) == 2 and hip_mid is not None:
            wrists_distance = _distance((wrists[0].x, wrists[0].y), (wrists[1].x, wrists[1].y))
            avg_wrist_y = sum(w.y for w in wrists) / 2
            in_torso_band = shoulder_mid[1] <= avg_wrist_y <= hip_mid[1]
            hands_close_together = wrists_distance <= shoulder_width * self._writing_ratio
            writing_or_reading_pose = in_torso_band and hands_close_together

        return PoseSignals(
            hands_near_phone=hands_near_phone,
            head_down=head_down,
            hands_hidden_below_hip=hands_hidden_below_hip,
            writing_or_reading_pose=writing_or_reading_pose,
            pose_detected=True,
        )


@dataclass
class _WindowEntry:
    timestamp: datetime
    signals: PoseSignals


class PoseWindowTracker:
    """
    Mantém, por câmera, uma janela deslizante de observações de pose (cada
    uma já condicionada a uma detecção de celular no mesmo frame) para medir
    a persistência dos sinais de contexto ao longo do tempo — o mesmo
    princípio de "não decidir com base em um único frame" usado para o
    celular também vale para o contexto de postura.
    """

    def __init__(self, window_seconds: float):
        self._window_seconds = window_seconds
        self._windows: Dict[str, Deque[_WindowEntry]] = {}

    def observe(self, camera_id: str, signals: PoseSignals, now: Optional[datetime] = None) -> None:
        now = now or datetime.now(timezone.utc)
        window = self._windows.setdefault(camera_id, deque())
        window.append(_WindowEntry(timestamp=now, signals=signals))
        cutoff = now - timedelta(seconds=self._window_seconds)
        while window and window[0].timestamp < cutoff:
            window.popleft()

    def persistence_ratio(self, camera_id: str) -> float:
        """Fração dos frames na janela em que houve sinal de mãos perto do celular OU cabeça baixa."""
        window = self._windows.get(camera_id)
        if not window:
            return 0.0
        flagged = sum(1 for e in window if e.signals.hands_near_phone or e.signals.head_down)
        return flagged / len(window)

    def reset(self, camera_id: str) -> None:
        self._windows.pop(camera_id, None)


class SuspicionScorer:
    """
    Combina confiança visual do celular + sinais de pose em uma pontuação
    explicável de 0 a 1. Isto NUNCA substitui a decisão humana — apenas
    ajuda a priorizar a fila de revisão.
    """

    def __init__(
        self,
        weight_phone_visual: float,
        weight_hand_proximity: float,
        weight_head_down: float,
        weight_hands_hidden: float,
        weight_persistence: float,
        weight_writing_reduction: float,
    ) -> None:
        self._w_phone = weight_phone_visual
        self._w_hand = weight_hand_proximity
        self._w_head = weight_head_down
        self._w_hidden = weight_hands_hidden
        self._w_persistence = weight_persistence
        self._w_writing = weight_writing_reduction

    def score(self, phone_confidence: float, pose_signals: PoseSignals) -> SuspicionBreakdown:
        reasons: List[str] = []
        raw_score = 0.0

        phone_contribution = self._w_phone * phone_confidence
        raw_score += phone_contribution
        reasons.append(
            f"detecção visual de celular: confiança {phone_confidence:.0%} (peso alto, contribuição {phone_contribution:.2f})"
        )

        if pose_signals.hands_near_phone:
            raw_score += self._w_hand
            reasons.append(f"mãos próximas ao celular (peso alto, contribuição {self._w_hand:.2f})")

        if pose_signals.head_down:
            raw_score += self._w_head
            reasons.append(f"cabeça direcionada para baixo/carteira (peso moderado, contribuição {self._w_head:.2f})")

        if pose_signals.hands_hidden_below_hip:
            raw_score += self._w_hidden
            reasons.append(
                f"mãos abaixo da linha do quadril, possivelmente fora da área visível (peso moderado, contribuição {self._w_hidden:.2f})"
            )

        if pose_signals.persistence_ratio > 0:
            persistence_contribution = self._w_persistence * pose_signals.persistence_ratio
            raw_score += persistence_contribution
            reasons.append(
                f"situação persistente em {pose_signals.persistence_ratio:.0%} dos frames recentes "
                f"(peso moderado, contribuição {persistence_contribution:.2f})"
            )

        if pose_signals.writing_or_reading_pose:
            raw_score -= self._w_writing
            reasons.append(
                f"postura compatível com escrita/leitura normal, mãos ocupadas (reduz suspeita, contribuição -{self._w_writing:.2f})"
            )

        final_score = max(0.0, min(1.0, raw_score))

        return SuspicionBreakdown(
            phone_confidence=phone_confidence,
            pose_signals=pose_signals,
            suspicion_score=final_score,
            reasons=reasons,
        )
