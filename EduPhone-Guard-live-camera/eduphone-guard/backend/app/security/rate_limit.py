"""Rate limiting básico para endpoints sensíveis, usando slowapi (baseado em token bucket por IP)."""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings

settings = get_settings()
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit_default])
