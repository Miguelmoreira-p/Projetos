from app.schemas.event import EventStatus, is_transition_allowed


def test_pending_can_go_to_confirmed():
    assert is_transition_allowed(EventStatus.PENDING, EventStatus.CONFIRMED)


def test_pending_can_go_to_false_positive():
    assert is_transition_allowed(EventStatus.PENDING, EventStatus.FALSE_POSITIVE)


def test_archived_is_terminal():
    assert not is_transition_allowed(EventStatus.ARCHIVED, EventStatus.PENDING)
    assert not is_transition_allowed(EventStatus.ARCHIVED, EventStatus.CONFIRMED)


def test_same_status_is_noop_allowed():
    assert is_transition_allowed(EventStatus.PENDING, EventStatus.PENDING)


def test_confirmed_can_be_reopened_to_pending():
    assert is_transition_allowed(EventStatus.CONFIRMED, EventStatus.PENDING)
