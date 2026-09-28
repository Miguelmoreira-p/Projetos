"""
Tracker de persistência por janela deslizante.

Não confiamos em uma única detecção de um único frame — isso gera excesso
de falsos positivos. Em vez disso, mantemos uma janela deslizante de
detecções recentes (por câmera) e só consideramos a condição "persistente"
satisfeita quando o número de frames com detecção válida, dentro da janela
de tempo configurada, atinge o mínimo exigido.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict

from app.schemas.detection import Detection


@dataclass
class _CameraWindow:
    window_seconds: float
    detections: Deque[Detection] = field(default_factory=deque)

    def add(self, detection: Detection) -> None:
        self.detections.append(detection)
        self._prune(detection.timestamp)

    def _prune(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.window_seconds)
        while self.detections and self.detections[0].timestamp < cutoff:
            self.detections.popleft()

    def count_in_window(self) -> int:
        return len(self.detections)

    def best_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return max(d.confidence for d in self.detections)


class PersistenceTracker:
    """
    Mantém, por câmera, uma janela deslizante de detecções acima do limiar
    de confiança e informa quando o critério de persistência é atingido.
    """

    def __init__(self, window_seconds: float, min_detection_frames: int, min_confidence: float):
        self._window_seconds = window_seconds
        self._min_detection_frames = min_detection_frames
        self._min_confidence = min_confidence
        self._windows: Dict[str, _CameraWindow] = {}

    def _window_for(self, camera_id: str) -> _CameraWindow:
        if camera_id not in self._windows:
            self._windows[camera_id] = _CameraWindow(window_seconds=self._window_seconds)
        return self._windows[camera_id]

    def register(self, detection: Detection) -> bool:
        """
        Registra uma detecção (já filtrada por confiança mínima pelo chamador
        ou aqui mesmo) e retorna True se o critério de persistência para a
        câmera correspondente foi atingido neste momento.
        """
        if detection.confidence < self._min_confidence:
            return False

        window = self._window_for(detection.camera_id)
        window.add(detection)
        return window.count_in_window() >= self._min_detection_frames

    def best_confidence(self, camera_id: str) -> float:
        return self._window_for(camera_id).best_confidence()

    def reset(self, camera_id: str) -> None:
        """Limpa a janela de uma câmera (usado após criar um evento, para evitar reprocessar as mesmas detecções)."""
        self._windows.pop(camera_id, None)
