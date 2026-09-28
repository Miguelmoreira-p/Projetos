"""
Buffer circular de frames em memória.

Mantém apenas os últimos N segundos de vídeo em memória (nunca em disco).
Quando um evento é disparado, o `recorder.py` usa este buffer para montar
um pequeno clipe (frames anteriores + frames capturados logo em seguida),
em vez de gravar o vídeo bruto continuamente.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np


@dataclass
class BufferedFrame:
    frame: np.ndarray
    timestamp: float  # segundos, relativo ao início da sessão de captura
    frame_number: int


class CircularFrameBuffer:
    """
    Buffer circular baseado em tempo (não em contagem fixa de frames), para
    se adaptar a variações de FPS entre câmeras.
    """

    def __init__(self, buffer_seconds: float, fps_estimate: float = 15.0):
        self._buffer_seconds = buffer_seconds
        # Estimativa inicial; ajustada dinamicamente conforme frames chegam.
        self._max_len = max(1, int(buffer_seconds * fps_estimate * 2))
        self._frames: Deque[BufferedFrame] = deque(maxlen=self._max_len)

    def append(self, frame: np.ndarray, timestamp: float, frame_number: int) -> None:
        self._frames.append(BufferedFrame(frame=frame.copy(), timestamp=timestamp, frame_number=frame_number))
        self._prune(timestamp)

    def _prune(self, now: float) -> None:
        cutoff = now - self._buffer_seconds
        while self._frames and self._frames[0].timestamp < cutoff:
            self._frames.popleft()

    def snapshot(self) -> List[BufferedFrame]:
        """Retorna uma cópia da lista de frames atualmente no buffer (mais antigo -> mais novo)."""
        return list(self._frames)

    def __len__(self) -> int:
        return len(self._frames)
