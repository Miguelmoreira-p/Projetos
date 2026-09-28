"""
Verificação de JWTs emitidos pelo Supabase Auth.

O frontend obtém o token via Supabase Auth (login por e-mail/senha ou
provedores configurados) e o envia no header `Authorization: Bearer <token>`.
O backend valida a assinatura usando o SUPABASE_JWT_SECRET (nunca confia
cegamente em um token não verificado) e extrai claims customizadas de papel
(role) e escola (school_id), que devem ser configuradas no Supabase via
Custom Access Token Hook ou tabela `app_metadata`.
"""
from __future__ import annotations

from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.config import get_settings
from app.schemas.user import AuthenticatedUser, Role

_bearer_scheme = HTTPBearer(auto_error=False)


class AuthError(HTTPException):
    def __init__(self, detail: str = "Não autenticado"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


def _decode_token(token: str) -> dict:
    settings = get_settings()
    if not settings.supabase_jwt_secret:
        raise RuntimeError("SUPABASE_JWT_SECRET não configurado no .env")
    try:
        return jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
        )
    except JWTError as exc:
        raise AuthError("Token inválido ou expirado") from exc


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AuthenticatedUser:
    if credentials is None:
        raise AuthError("Credenciais ausentes")

    payload = _decode_token(credentials.credentials)

    user_id = payload.get("sub")
    email = payload.get("email")
    app_metadata = payload.get("app_metadata", {}) or {}
    role_raw = app_metadata.get("role")
    school_id_raw = app_metadata.get("school_id")

    if not user_id or not email or not role_raw:
        raise AuthError("Token não contém as claims necessárias (role, email)")

    try:
        role = Role(role_raw)
    except ValueError as exc:
        raise AuthError(f"Papel de usuário desconhecido: {role_raw}") from exc

    return AuthenticatedUser(
        id=UUID(user_id),
        email=email,
        role=role,
        school_id=UUID(school_id_raw) if school_id_raw else None,
    )
