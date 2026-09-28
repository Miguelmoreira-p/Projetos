from unittest.mock import MagicMock

import pytest

from app.services import retention_service


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, rows_to_return):
        self._rows_to_return = rows_to_return
        self.updates = []

    def select(self, *_args, **_kwargs):
        return self

    def lt(self, *_args, **_kwargs):
        return self

    def neq(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def execute(self):
        return _FakeResponse(self._rows_to_return)

    def update(self, values):
        self.updates.append(values)
        return self


class _FakeTable:
    def __init__(self, rows_to_return):
        self._query = _FakeQuery(rows_to_return)

    def select(self, *args, **kwargs):
        return self._query.select(*args, **kwargs)

    def update(self, values):
        return self._query.update(values)


class _FakeClient:
    def __init__(self, rows_to_return):
        self._table = _FakeTable(rows_to_return)

    def table(self, _name):
        return self._table


@pytest.fixture
def fake_rows():
    return [
        {"id": "event-1", "video_path": "school-a/event-1.mp4"},
        {"id": "event-2", "video_path": None},
    ]


def test_sweep_expired_events_processes_all_rows(monkeypatch, fake_rows):
    fake_client = _FakeClient(fake_rows)
    monkeypatch.setattr(retention_service, "get_service_client", lambda: fake_client)
    monkeypatch.setattr(retention_service.storage_service, "delete_clip", MagicMock())
    monkeypatch.setattr(retention_service, "record_audit_log", MagicMock())

    processed = retention_service.sweep_expired_events()

    assert processed == 2
    retention_service.storage_service.delete_clip.assert_called_once_with("school-a/event-1.mp4")
    assert retention_service.record_audit_log.call_count == 2


def test_sweep_handles_storage_deletion_failure_gracefully(monkeypatch, fake_rows):
    fake_client = _FakeClient(fake_rows)
    monkeypatch.setattr(retention_service, "get_service_client", lambda: fake_client)

    def _boom(_path):
        raise RuntimeError("storage indisponível")

    monkeypatch.setattr(retention_service.storage_service, "delete_clip", _boom)
    monkeypatch.setattr(retention_service, "record_audit_log", MagicMock())

    # Não deve levantar exceção mesmo se a exclusão no storage falhar.
    processed = retention_service.sweep_expired_events()
    assert processed == 2
