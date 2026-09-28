"""Schemas relacionados à detecção bruta de objetos (saída do modelo de visão)."""
from __future__ import annotations

from datetime import datetime
from typing import Tuple

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class Detection(BaseModel):
    """
    Representa uma única detecção de objeto em um frame.

    IMPORTANTE: este objeto NUNCA deve conter identificadores de pessoas
    (nome, rosto, matrícula). Ele descreve apenas o objeto detectado
    ("cell_phone") e sua localização/; confiança no frame.
    """

    detected_class: str = Field(..., description="Classe detectada, ex: 'cell_phone'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    timestamp: datetime
    frame_number: int = Field(..., ge=0)
    camera_id: str

    def to_metadata_dict(self) -> dict:
        """Representação serializável para armazenar em detection_metadata (JSONB)."""
        return {
            "class": self.detected_class,
            "confidence": round(self.confidence, 4),
            "bbox": [self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2],
            "timestamp": self.timestamp.isoformat(),
            "frame": self.frame_number,
        }
