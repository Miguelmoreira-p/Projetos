import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.auth.jwt import get_current_user
from app.main import app
from app.schemas.user import AuthenticatedUser, Role
from app.services import event_service


@pytest.fixture
def client():
    return TestClient(app)


def _override_user(role: Role, school_id=None):
    def _dep():
        return AuthenticatedUser(id=uuid.uuid4(), email="user@example.com", role=role, school_id=school_id)

    return _dep


def test_health_check_is_public(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_events_requires_authentication(client):
    response = client.get("/events")
    assert response.status_code in (401, 403)


def test_events_list_returns_data_for_authenticated_user(client, monkeypatch):
    app.dependency_overrides[get_current_user] = _override_user(Role.ADMIN)
    monkeypatch.setattr(event_service, "list_events", MagicMock(return_value=[]))

    # A rota importa event_service diretamente; garantimos o patch no módulo usado pela rota.
    from app.api import events as events_routes

    monkeypatch.setattr(events_routes.event_service, "list_events", MagicMock(return_value=[]))

    response = client.get("/events")
    assert response.status_code == 200
    assert response.json() == []

    app.dependency_overrides.clear()


def test_camera_creation_forbidden_for_reviewer(client):
    app.dependency_overrides[get_current_user] = _override_user(Role.REVIEWER)

    response = client.post(
        "/cameras",
        json={
            "school_id": str(uuid.uuid4()),
            "name": "Sala 8",
            "location": "Bloco A",
        },
    )
    assert response.status_code == 403

    app.dependency_overrides.clear()
