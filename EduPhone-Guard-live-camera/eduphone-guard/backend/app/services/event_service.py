"""Camada de serviço para eventos: criação, listagem, e revisão humana."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import HTTPException, status

from app.config import get_settings
from app.database import get_service_client
from app.schemas.event import EventCreate, EventOut, EventReviewCreate, EventStatus, is_transition_allowed
from app.schemas.user import AuthenticatedUser, Role
from app.services.audit_service import record_audit_log


def create_event(payload: EventCreate) -> EventOut:
    """
    Cria um evento a partir de um EventTrigger da pipeline de visão.
    Nunca chamado a partir de um único frame — a persistência/cooldown já
    foram validados em app.vision.event_detector antes de chegar aqui.
    """
    settings = get_settings()
    retention_days = payload.retention_days_override or settings.retention_days
    retention_until = (
        datetime.now(timezone.utc) + timedelta(days=retention_days) if retention_days > 0 else None
    )

    row = {
        "school_id": str(payload.school_id),
        "camera_id": str(payload.camera_id),
        "detected_at": payload.detected_at.isoformat(),
        "confidence": payload.confidence,
        "status": EventStatus.PENDING.value,
        "video_path": payload.video_path,
        "retention_until": retention_until.isoformat() if retention_until else None,
        "detection_metadata": payload.detection_metadata,
    }

    response = get_service_client().table("events").insert(row).execute()
    created = response.data[0]

    record_audit_log(
        user_id=None,
        action="event_created",
        resource_type="event",
        resource_id=created["id"],
        metadata={"camera_id": str(payload.camera_id), "confidence": payload.confidence},
    )

    return EventOut(**created)


def list_events(user: AuthenticatedUser, status_filter: Optional[EventStatus] = None) -> List[EventOut]:
    client = get_service_client()
    query = client.table("events").select("*").order("detected_at", desc=True)

    if user.role != Role.ADMIN:
        if user.school_id is None:
            return []
        query = query.eq("school_id", str(user.school_id))

    if status_filter is not None:
        query = query.eq("status", status_filter.value)

    response = query.execute()
    return [EventOut(**row) for row in response.data]


def get_event(user: AuthenticatedUser, event_id: UUID) -> EventOut:
    client = get_service_client()
    response = client.table("events").select("*").eq("id", str(event_id)).limit(1).execute()
    if not response.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evento não encontrado")

    event = EventOut(**response.data[0])
    if user.role != Role.ADMIN and str(event.school_id) != str(user.school_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Acesso negado")

    return event


def review_event(user: AuthenticatedUser, event_id: UUID, payload: EventReviewCreate) -> EventOut:
    event = get_event(user, event_id)

    if not is_transition_allowed(event.status, payload.status):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Transição de {event.status.value} para {payload.status.value} não é permitida.",
        )

    client = get_service_client()

    update_response = (
        client.table("events")
        .update({"status": payload.status.value})
        .eq("id", str(event_id))
        .execute()
    )

    review_row = {
        "event_id": str(event_id),
        "reviewer_id": str(user.id),
        "status": payload.status.value,
        "notes": payload.notes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    client.table("event_reviews").insert(review_row).execute()

    record_audit_log(
        user_id=user.id,
        action="event_reviewed",
        resource_type="event",
        resource_id=str(event_id),
        metadata={"new_status": payload.status.value},
    )

    return EventOut(**update_response.data[0])
