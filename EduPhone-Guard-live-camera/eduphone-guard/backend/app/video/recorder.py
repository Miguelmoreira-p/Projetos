"""
Monta e persiste apenas o pequeno trecho de vídeo necessário para revisão
humana de um evento — nunca o vídeo bruto contínuo.

Fluxo:
1. No momento do gatilho, pega o snapshot do CircularFrameBuffer (contém os
   últimos `buffer_seconds` segundos).
2. Continua coletando frames por mais `post_event_seconds` segundos.
3. Grava tudo em um único arquivo de vídeo temporário local, em
   `clip_output_dir`, com nome baseado em UUID do evento.
4. Retorna o caminho do arquivo para ser enviado ao Supabase Storage (bucket
   privado) pelo camada de serviço.

Nenhum outro frame fora dessa janela é persistido em disco.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np

from app.video.buffer import BufferedFrame

logger = logging.getLogger("eduphone_guard.video.recorder")


class ClipRecorder:
    def __init__(self, output_dir: str, max_width: int = 960, fps: float = 15.0):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._max_width = max_width
        self._fps = fps

    def _resize_if_needed(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= self._max_width:
            return frame
        scale = self._max_width / float(width)
        return cv2.resize(frame, (self._max_width, int(height * scale)))

    def write_clip(self, event_id: uuid.UUID, pre_frames: List[BufferedFrame], post_frames: List[np.ndarray]) -> Path:
        """
        Grava um único arquivo MP4 combinando os frames anteriores ao evento
        (do buffer circular) e os frames posteriores capturados após o
        gatilho. Retorna o caminho local do arquivo gerado.
        """
        if not pre_frames and not post_frames:
            raise ValueError("Nenhum frame disponível para montar o clipe do evento.")

        sample_frame = pre_frames[0].frame if pre_frames else post_frames[0]
        sample_frame = self._resize_if_needed(sample_frame)
        height, width = sample_frame.shape[:2]

        output_path = self._output_dir / f"event_{event_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, self._fps, (width, height))

        try:
            for buffered in pre_frames:
                writer.write(self._resize_if_needed(buffered.frame))
            for frame in post_frames:
                writer.write(self._resize_if_needed(frame))
        finally:
            writer.release()

        logger.info(
            "clip_written event_id=%s path=%s frames=%d",
            event_id,
            output_path,
            len(pre_frames) + len(post_frames),
        )
        return output_path

    def delete_clip(self, path: Path) -> None:
        """Remove um clipe local (ex.: após upload bem-sucedido para o Storage, ou na varredura de retenção)."""
        try:
            if path.exists():
                path.unlink()
        except OSError:
            logger.exception("failed_to_delete_clip path=%s", path)
