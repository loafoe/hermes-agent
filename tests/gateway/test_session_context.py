"""Tests for gateway/session_context.py's per-task ContextVar session state."""


def test_get_session_mcp_jwt_defaults_to_none():
    from gateway.session_context import get_session_mcp_jwt, reset_session_vars

    reset_session_vars()
    assert get_session_mcp_jwt() is None


def test_set_session_vars_binds_mcp_jwt():
    from gateway.session_context import (
        clear_session_vars,
        get_session_mcp_jwt,
        set_session_vars,
    )

    tokens = set_session_vars(mcp_jwt="eyJ.example.jwt")
    try:
        assert get_session_mcp_jwt() == "eyJ.example.jwt"
    finally:
        clear_session_vars(tokens)


def test_clear_session_vars_clears_mcp_jwt():
    from gateway.session_context import (
        clear_session_vars,
        get_session_mcp_jwt,
        set_session_vars,
    )

    tokens = set_session_vars(mcp_jwt="eyJ.example.jwt")
    clear_session_vars(tokens)
    assert get_session_mcp_jwt() is None


def test_set_session_vars_without_mcp_jwt_defaults_to_none():
    """Existing callers that don't pass mcp_jwt must not break (regression guard)."""
    from gateway.session_context import (
        clear_session_vars,
        get_session_mcp_jwt,
        set_session_vars,
    )

    tokens = set_session_vars(platform="api_server")
    try:
        assert get_session_mcp_jwt() is None
    finally:
        clear_session_vars(tokens)


def test_mcp_jwt_does_not_fall_back_to_os_environ(monkeypatch):
    """Unlike get_session_env(), get_session_mcp_jwt() must never read
    os.environ — a stale process-global var must not leak a JWT across
    sessions (this is exactly the class of bug this module's docstring
    describes for the old os.environ-based design)."""
    import os
    from gateway.session_context import get_session_mcp_jwt, reset_session_vars

    monkeypatch.setenv("HERMES_SESSION_MCP_JWT", "should-never-be-read")
    reset_session_vars()
    assert get_session_mcp_jwt() is None
    os.environ.pop("HERMES_SESSION_MCP_JWT", None)
