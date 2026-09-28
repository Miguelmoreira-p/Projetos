import uuid

import pytest
from jose import jwt

from app.auth.jwt import AuthError, _decode_token, get_current_user
from app.config import get_settings
from fastapi.security import HTTPAuthorizationCredentials


SECRET = "test-secret-for-unit-tests-only"


def _make_token(**overrides) -> str:
    payload = {
        "sub": str(uuid.uuid4()),
        "email": "user@example.com",
        "aud": "authenticated",
        "app_metadata": {"role": "ADMIN", "school_id": None},
    }
    payload.update(overrides)
    return jwt.encode(payload, SECRET, algorithm="HS256")


@pytest.fixture(autouse=True)
def _configure_secret(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("SUPABASE_JWT_SECRET", SECRET)
    yield
    get_settings.cache_clear()


def test_decode_valid_token():
    token = _make_token()
    payload = _decode_token(token)
    assert payload["email"] == "user@example.com"


def test_decode_invalid_signature_raises():
    bad_token = jwt.encode({"sub": "x", "aud": "authenticated"}, "wrong-secret", algorithm="HS256")
    with pytest.raises(AuthError):
        _decode_token(bad_token)


def test_get_current_user_missing_credentials_raises():
    with pytest.raises(AuthError):
        get_current_user(credentials=None)


def test_get_current_user_builds_authenticated_user():
    token = _make_token(app_metadata={"role": "DIRECTOR", "school_id": str(uuid.uuid4())})
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    user = get_current_user(credentials=creds)
    assert user.role.value == "DIRECTOR"
    assert user.school_id is not None
