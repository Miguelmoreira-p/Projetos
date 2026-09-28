import uuid

import pytest
from fastapi import HTTPException

from app.schemas.user import AuthenticatedUser, Role
from app.security.rbac import require_roles, require_same_school


def _user(role: Role, school_id=None) -> AuthenticatedUser:
    return AuthenticatedUser(id=uuid.uuid4(), email="user@example.com", role=role, school_id=school_id)


def test_require_roles_allows_matching_role():
    dependency = require_roles(Role.ADMIN)
    user = _user(Role.ADMIN)
    assert dependency(user) is user


def test_require_roles_blocks_non_matching_role():
    dependency = require_roles(Role.ADMIN)
    user = _user(Role.REVIEWER)
    with pytest.raises(HTTPException) as exc_info:
        dependency(user)
    assert exc_info.value.status_code == 403


def test_require_same_school_allows_admin_any_school():
    admin = _user(Role.ADMIN)
    require_same_school(admin, uuid.uuid4())  # não deve lançar


def test_require_same_school_blocks_other_school():
    school_a = uuid.uuid4()
    school_b = uuid.uuid4()
    director = _user(Role.DIRECTOR, school_id=school_a)
    with pytest.raises(HTTPException) as exc_info:
        require_same_school(director, school_b)
    assert exc_info.value.status_code == 403


def test_require_same_school_allows_same_school():
    school = uuid.uuid4()
    director = _user(Role.DIRECTOR, school_id=school)
    require_same_school(director, school)  # não deve lançar
