"""
Wrapper fino sobre o client Python do Supabase.

Duas instâncias são expostas:
- `get_public_client()`: usa a anon key, respeita Row Level Security (RLS).
  Usada quando queremos que o próprio Postgres aplique as políticas de
  acesso baseadas no usuário autenticado.
- `get_service_client()`: usa a service role key (bypassa RLS). Deve ser
  usada SOMENTE em operações de backend confiáveis (ex.: inserir evento
  gerado pela pipeline de visão, rotina de retenção). NUNCA exponha a
  service role key ao frontend.
"""
from __future__ import annotations

from functools import lru_cache

from supabase import Client, create_client

from app.config import get_settings


@lru_cache
def get_service_client() -> Client:
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY não configurados. "
            "Preencha o arquivo .env a partir de .env.example."
        )
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


@lru_cache
def get_public_client() -> Client:
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_anon_key:
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_ANON_KEY não configurados. "
            "Preencha o arquivo .env a partir de .env.example."
        )
    return create_client(settings.supabase_url, settings.supabase_anon_key)
