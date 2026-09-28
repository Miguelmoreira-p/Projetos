from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.jwt import get_current_user
from app.database import get_service_client
from app.schemas.camera import CameraCreate, CameraOut
from app.schemas.user import AuthenticatedUser, Role
from app.security.rbac import require_roles
from app.services.audit_service import record_audit_log

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("", response_model=List[CameraOut])
def list_cameras(user: AuthenticatedUser = Depends(get_current_user)):
    client = get_service_client()
    query = client.table("cameras").select("id, school_id, name, location, active, created_at")
    if user.role != Role.ADMIN:
        if user.school_id is None:
            return []
        query = query.eq("school_id", str(user.school_id))
    response = query.execute()
    return [CameraOut(**row) for row in response.data]


@router.post("", response_model=CameraOut, status_code=status.HTTP_201_CREATED)
def create_camera(
    payload: CameraCreate,
    user: AuthenticatedUser = Depends(require_roles(Role.ADMIN)),
):
    """
    Cria um cadastro de câmera. Nota: `stream_url` é armazenado mas nunca
    retornado neste schema de saída (CameraOut) — não deve ser exposto ao
    frontend de forma ampla, apenas usado internamente pelo serviço de
    captura de vídeo.
    """
    client = get_service_client()
    row = payload.model_dump(mode="json")
    response = client.table("cameras").insert(row).execute()
    created = response.data[0]

    record_audit_log(
        user_id=user.id,
        action="camera_created",
        resource_type="camera",
        resource_id=created["id"],
        metadata={"name": payload.name},
    )
    return CameraOut(**created)


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def deactivate_camera(camera_id: UUID, user: AuthenticatedUser = Depends(require_roles(Role.ADMIN))):
    client = get_service_client()
    response = client.table("cameras").update({"active": False}).eq("id", str(camera_id)).execute()
    if not response.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Câmera não encontrada")

    record_audit_log(
        user_id=user.id,
        action="camera_deactivated",
        resource_type="camera",
        resource_id=str(camera_id),
        metadata={},
    )
