"""Registro de logs de auditoria para ações administrativas relevantes."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID

from app.database import get_service_client

logger = logging.getLogger("eduphone_guard.audit")

SENSITIVE_KEYS = {"password", "token", "secret", "key", "authorization"}


def _sanitize(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Remove qualquer chave que pareça sensível antes de persistir/logar."""
    return {k: v for k, v in metadata.items() if k.lower() not in SENSITIVE_KEYS}


def record_audit_log(
    user_id: Optional[UUID],
    action: str,
    resource_type: str,
    resource_id: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    clean_metadata = _sanitize(metadata or {})
    row = {
        "user_id": str(user_id) if user_id else None,
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": clean_metadata,
    }
    try:
        get_service_client().table("audit_logs").insert(row).execute()
    except Exception:
        # Falha ao registrar auditoria não deve derrubar a operação principal,
        # mas deve ficar visível nos logs da aplicação.
        logger.exception("failed_to_write_audit_log action=%s resource_type=%s", action, resource_type)

    logger.info("audit action=%s resource_type=%s resource_id=%s", action, resource_type, resource_id)
