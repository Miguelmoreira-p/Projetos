from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class EventStatus(str, Enum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    ARCHIVED = "ARCHIVED"


# Transições de estado permitidas (origem -> destinos válidos).
ALLOWED_TRANSITIONS: Dict[EventStatus, List[EventStatus]] = {
    EventStatus.PENDING: [EventStatus.CONFIRMED, EventStatus.FALSE_POSITIVE, EventStatus.ARCHIVED],
    EventStatus.CONFIRMED: [EventStatus.ARCHIVED, EventStatus.PENDING],
    EventStatus.FALSE_POSITIVE: [EventStatus.ARCHIVED, EventStatus.PENDING],
    EventStatus.ARCHIVED: [],
}


def is_transition_allowed(current: EventStatus, target: EventStatus) -> bool:
    if current == target:
        return True
    return target in ALLOWED_TRANSITIONS.get(current, [])


class EventCreate(BaseModel):
    school_id: UUID
    camera_id: UUID
    detected_at: datetime
    confidence: float = Field(..., ge=0.0, le=1.0)
    video_path: Optional[str] = None
    detection_metadata: Dict[str, Any] = Field(default_factory=dict)
    retention_days_override: Optional[int] = None


class EventOut(BaseModel):
    id: UUID
    school_id: UUID
    camera_id: UUID
    created_at: datetime
    detected_at: datetime
    confidence: float
    status: EventStatus
    video_path: Optional[str] = None
    retention_until: Optional[datetime] = None
    detection_metadata: Dict[str, Any] = Field(default_factory=dict)


class EventReviewCreate(BaseModel):
    status: EventStatus
    notes: Optional[str] = Field(default=None, max_length=2000)


class EventReviewOut(BaseModel):
    id: UUID
    event_id: UUID
    reviewer_id: UUID
    status: EventStatus
    notes: Optional[str] = None
    created_at: datetime
