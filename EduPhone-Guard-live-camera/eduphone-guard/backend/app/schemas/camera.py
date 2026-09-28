from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class CameraCreate(BaseModel):
    school_id: UUID
    name: str = Field(..., max_length=120)
    location: str = Field(..., max_length=200)
    stream_url: Optional[str] = Field(
        default=None,
        description="URL/índice do dispositivo de vídeo. Nunca deve ser exposta ao frontend público.",
    )
    active: bool = True


class CameraOut(BaseModel):
    id: UUID
    school_id: UUID
    name: str
    location: str
    active: bool
    created_at: datetime


class SystemSettingsOut(BaseModel):
    min_confidence: float
    min_detection_frames: int
    window_seconds: float
    cooldown_seconds: float
    buffer_seconds: float
    post_event_seconds: float
    retention_days: int
    simulation_mode: bool


class SystemSettingsUpdate(BaseModel):
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_detection_frames: Optional[int] = Field(default=None, ge=1)
    window_seconds: Optional[float] = Field(default=None, gt=0)
    cooldown_seconds: Optional[float] = Field(default=None, ge=0)
    retention_days: Optional[int] = Field(default=None, ge=0)
