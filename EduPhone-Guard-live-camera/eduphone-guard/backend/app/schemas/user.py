from __future__ import annotations

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, EmailStr


class Role(str, Enum):
    ADMIN = "ADMIN"
    DIRECTOR = "DIRECTOR"
    REVIEWER = "REVIEWER"


class AuthenticatedUser(BaseModel):
    """Usuário autenticado, derivado do JWT emitido pelo Supabase Auth."""

    id: UUID
    email: EmailStr
    role: Role
    school_id: UUID | None = None
