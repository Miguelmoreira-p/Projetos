"""
Controle de acesso baseado em papel (RBAC), aplicado no backend.

IMPORTANTE: autorização real acontece aqui (server-side) e também via Row
Level Security no Postgres (ver sql/schema.sql). O frontend pode esconder
botões por conveniência de UX, mas isso nunca é a barreira de segurança.
"""
from __future__ import annotations

from typing import Callable, Iterable

from fastapi import Depends, HTTPException, status

from app.auth.jwt import get_current_user
from app.schemas.user import AuthenticatedUser, Role


def require_roles(*allowed_roles: Role) -> Callable[[AuthenticatedUser], AuthenticatedUser]:
    def _dependency(user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Papel '{user.role.value}' não tem permissão para esta ação.",
            )
        return user

    return _dependency


def require_same_school(user: AuthenticatedUser, resource_school_id) -> None:
    """Garante que um usuário (não-ADMIN) só acesse recursos da própria escola."""
    if user.role == Role.ADMIN:
        return
    if user.school_id is None or str(user.school_id) != str(resource_school_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acesso negado a recursos de outra escola.",
        )
