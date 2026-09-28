"""
Integração com Supabase Storage.

Regras invioláveis:
- O bucket de mídia é SEMPRE privado.
- Nenhum vídeo é servido por URL pública direta.
- Acesso a um clipe específico só é concedido depois de uma verificação de
  permissão no backend, através de uma URL assinada com TTL curto.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.config import get_settings
from app.database import get_service_client

logger = logging.getLogger("eduphone_guard.services.storage")


def upload_clip(local_path: Path, storage_path: str) -> None:
    settings = get_settings()
    client = get_service_client()
    with open(local_path, "rb") as f:
        client.storage.from_(settings.supabase_media_bucket).upload(
            path=storage_path,
            file=f,
            file_options={"content-type": "video/mp4", "upsert": "true"},
        )
    logger.info("clip_uploaded storage_path=%s", storage_path)


def create_signed_url(storage_path: str) -> str:
    settings = get_settings()
    client = get_service_client()
    result = client.storage.from_(settings.supabase_media_bucket).create_signed_url(
        path=storage_path,
        expires_in=settings.signed_url_ttl_seconds,
    )
    signed_url = result.get("signedURL") or result.get("signed_url")
    if not signed_url:
        raise RuntimeError(f"Falha ao gerar URL assinada para {storage_path!r}: {result!r}")
    return signed_url


def delete_clip(storage_path: str) -> None:
    settings = get_settings()
    client = get_service_client()
    client.storage.from_(settings.supabase_media_bucket).remove([storage_path])
    logger.info("clip_deleted_from_storage storage_path=%s", storage_path)
