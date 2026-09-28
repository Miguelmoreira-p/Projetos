from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends

from app.auth.jwt import get_current_user
from app.schemas.event import EventOut, EventReviewCreate, EventStatus
from app.schemas.user import AuthenticatedUser, Role
from app.security.rbac import require_roles
from app.services import event_service, storage_service

router = APIRouter(prefix="/events", tags=["events"])


@router.get("", response_model=List[EventOut])
def list_events(
    status_filter: Optional[EventStatus] = None,
    user: AuthenticatedUser = Depends(get_current_user),
):
    return event_service.list_events(user, status_filter=status_filter)


@router.get("/{event_id}", response_model=EventOut)
def get_event(event_id: UUID, user: AuthenticatedUser = Depends(get_current_user)):
    return event_service.get_event(user, event_id)


@router.get("/{event_id}/media-url")
def get_event_media_url(event_id: UUID, user: AuthenticatedUser = Depends(get_current_user)):
    """
    Retorna uma URL assinada de curta duração para o clipe do evento.
    Nunca retorna uma URL pública permanente.
    """
    event = event_service.get_event(user, event_id)
    if not event.video_path:
        return {"url": None}
    return {"url": storage_service.create_signed_url(event.video_path)}


@router.post("/{event_id}/review", response_model=EventOut)
def review_event(
    event_id: UUID,
    payload: EventReviewCreate,
    user: AuthenticatedUser = Depends(require_roles(Role.ADMIN, Role.DIRECTOR, Role.REVIEWER)),
):
    return event_service.review_event(user, event_id, payload)
