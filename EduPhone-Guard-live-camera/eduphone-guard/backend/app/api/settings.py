from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import get_settings
from app.schemas.camera import SystemSettingsOut, SystemSettingsUpdate
from app.schemas.user import AuthenticatedUser, Role
from app.security.rbac import require_roles
from app.services.audit_service import record_audit_log

router = APIRouter(prefix="/settings", tags=["settings"])

# Nota de implementação: no MVP, as configurações operacionais (limiares de
# detecção, retenção, etc.) vivem em variáveis de ambiente (app/config.py),
# recarregadas no processo de visão computacional. Este endpoint expõe/edita
# os valores atuais através da tabela `system_settings` no Supabase para
# permitir ajuste pela equipe da escola sem reiniciar o serviço; uma versão
# futura deve fazer o pipeline de visão observar essa tabela em tempo real.


@router.get("", response_model=SystemSettingsOut)
def get_current_settings(user: AuthenticatedUser = Depends(require_roles(Role.ADMIN, Role.DIRECTOR))):
    settings = get_settings()
    return SystemSettingsOut(
        min_confidence=settings.min_confidence,
        min_detection_frames=settings.min_detection_frames,
        window_seconds=settings.window_seconds,
        cooldown_seconds=settings.cooldown_seconds,
        buffer_seconds=settings.buffer_seconds,
        post_event_seconds=settings.post_event_seconds,
        retention_days=settings.retention_days,
        simulation_mode=settings.simulation_mode,
    )


@router.patch("", response_model=SystemSettingsOut)
def update_settings(
    payload: SystemSettingsUpdate,
    user: AuthenticatedUser = Depends(require_roles(Role.ADMIN)),
):
    from app.database import get_service_client

    client = get_service_client()
    updates = {k: v for k, v in payload.model_dump(exclude_none=True).items()}
    if updates:
        client.table("system_settings").upsert({"id": "default", **updates}).execute()
        record_audit_log(
            user_id=user.id,
            action="settings_updated",
            resource_type="system_settings",
            resource_id="default",
            metadata=updates,
        )

    settings = get_settings()
    merged = settings.model_dump()
    merged.update(updates)
    return SystemSettingsOut(
        min_confidence=merged["min_confidence"],
        min_detection_frames=merged["min_detection_frames"],
        window_seconds=merged["window_seconds"],
        cooldown_seconds=merged["cooldown_seconds"],
        buffer_seconds=merged["buffer_seconds"],
        post_event_seconds=merged["post_event_seconds"],
        retention_days=merged["retention_days"],
        simulation_mode=merged["simulation_mode"],
    )
