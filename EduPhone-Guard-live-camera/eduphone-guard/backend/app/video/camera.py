"""
Abstração de fonte de vídeo: webcam (índice de dispositivo) ou arquivo MP4
(usado no modo de simulação). Processamento sempre local — não faz upload
de vídeo bruto para qualquer lugar.
"""
from __future__ import annotations

import logging
import time
from typing import Iterator, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("eduphone_guard.video.camera")


class VideoSource:
    """
    Encapsula cv2.VideoCapture, aceitando tanto um índice de webcam (int)
    quanto um caminho de arquivo de vídeo (str), e expõe um gerador de
    frames com timestamp relativo e número de frame.
    """

    def __init__(self, source: Union[int, str]):
        self._source = source
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {source!r}")

        self.fps = self._capture.get(cv2.CAP_PROP_FPS) or 15.0
        if self.fps <= 0:
            self.fps = 15.0

    def frames(self) -> Iterator[Tuple[np.ndarray, float, int]]:
        """Gera tuplas (frame, timestamp_relativo_segundos, frame_number)."""
        frame_number = 0
        start = time.monotonic()
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            elapsed = time.monotonic() - start
            yield frame, elapsed, frame_number
            frame_number += 1

    def release(self) -> None:
        self._capture.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
