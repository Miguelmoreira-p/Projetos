"""
Rotina de retenção: encontra eventos expirados (retention_until no passado),
remove o vídeo do Storage e anonimiza/remove metadados, registrando a ação
no audit log.

Não assuma que o valor padrão de RETENTION_DAYS é juridicamente adequado —
é apenas um padrão técnico configurável (ver README, seção Privacidade).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.database import get_service_client
from app.services import storage_service
from app.services.audit_service import record_audit_log

logger = logging.getLogger("eduphone_guard.services.retention")


def sweep_expired_events() -> int:
    """Executa uma varredura de retenção. Retorna o número de eventos processados."""
    client = get_service_client()
    now_iso = datetime.now(timezone.utc).isoformat()

    response = (
        client.table("events")
        .select("id, video_path")
        .lt("retention_until", now_iso)
        .neq("status", "ARCHIVED")
        .execute()
    )

    processed = 0
    for row in response.data:
        event_id = row["id"]
        video_path = row.get("video_path")

        if video_path:
            try:
                storage_service.delete_clip(video_path)
            except Exception:
                logger.exception("failed_to_delete_clip_during_retention event_id=%s", event_id)

        client.table("events").update(
            {"status": "ARCHIVED", "video_path": None, "detection_metadata": {}}
        ).eq("id", event_id).execute()

        record_audit_log(
            user_id=None,
            action="event_deleted_retention_expired",
            resource_type="event",
            resource_id=event_id,
            metadata={},
        )
        processed += 1

    logger.info("retention_sweep_completed processed=%d", processed)
    return processed
