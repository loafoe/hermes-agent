"""Tests for JWT Bearer-token forwarding to MCP HTTP/SSE tool calls.

Mirrors the design in
docs/superpowers/specs/2026-07-16-mcp-jwt-forwarding-design.md: a caller's
JWT is captured per-request by the API server, propagated via a
ContextVar + the existing _pending_call_context-style thread bridge, and
injected into outgoing MCP requests by _ForwardedJWTAuth — the same
httpx.Auth seam HermesMCPOAuthProvider already uses for OAuth.
"""

import httpx
import pytest


def test_pending_mcp_jwt_defaults_to_none():
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("srv")
    assert server._pending_mcp_jwt is None


def test_forwarded_jwt_auth_sets_authorization_header():
    from tools.mcp_tool import MCPServerTask, _ForwardedJWTAuth

    server = MCPServerTask("srv")
    server._pending_mcp_jwt = "eyJhbGciOiJIUzI1NiJ9.example.sig"
    auth = _ForwardedJWTAuth(server)

    request = httpx.Request("POST", "https://example.com/mcp")
    flow = auth.auth_flow(request)
    outgoing = next(flow)

    assert outgoing.headers["Authorization"] == "Bearer eyJhbGciOiJIUzI1NiJ9.example.sig"


def test_forwarded_jwt_auth_no_op_when_jwt_absent():
    from tools.mcp_tool import MCPServerTask, _ForwardedJWTAuth

    server = MCPServerTask("srv")
    assert server._pending_mcp_jwt is None
    auth = _ForwardedJWTAuth(server)

    request = httpx.Request("POST", "https://example.com/mcp")
    flow = auth.auth_flow(request)
    outgoing = next(flow)

    assert "Authorization" not in outgoing.headers


def test_forwarded_jwt_auth_overwrites_static_authorization_header():
    """Dynamic forwarded JWT wins over a static config 'headers' value —
    matches picoclaw's documented dynamic-overrides-static precedent."""
    from tools.mcp_tool import MCPServerTask, _ForwardedJWTAuth

    server = MCPServerTask("srv")
    server._pending_mcp_jwt = "forwarded-token"
    auth = _ForwardedJWTAuth(server)

    request = httpx.Request(
        "POST", "https://example.com/mcp",
        headers={"Authorization": "Bearer static-configured-token"},
    )
    flow = auth.auth_flow(request)
    outgoing = next(flow)

    assert outgoing.headers["Authorization"] == "Bearer forwarded-token"


def test_forwarded_jwt_auth_reads_fresh_value_each_call():
    """The class holds a server reference, not a snapshot — successive
    calls on the same MCPServerTask can carry different callers' JWTs."""
    from tools.mcp_tool import MCPServerTask, _ForwardedJWTAuth

    server = MCPServerTask("srv")
    auth = _ForwardedJWTAuth(server)

    server._pending_mcp_jwt = "token-a"
    request_a = httpx.Request("POST", "https://example.com/mcp")
    outgoing_a = next(auth.auth_flow(request_a))
    assert outgoing_a.headers["Authorization"] == "Bearer token-a"

    server._pending_mcp_jwt = "token-b"
    request_b = httpx.Request("POST", "https://example.com/mcp")
    outgoing_b = next(auth.auth_flow(request_b))
    assert outgoing_b.headers["Authorization"] == "Bearer token-b"


def test_make_tool_handler_snapshots_session_mcp_jwt_during_call(monkeypatch, tmp_path):
    """_make_tool_handler's _call() must snapshot get_session_mcp_jwt() into
    server._pending_mcp_jwt before session.call_tool(), and clear it after —
    mirroring the existing _pending_call_context bridge for elicitation."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import json
    from unittest.mock import MagicMock

    from tools import mcp_tool
    from tools.mcp_tool import MCPServerTask, _make_tool_handler
    from gateway.session_context import clear_session_vars, set_session_vars

    server = MCPServerTask("srv")
    observed = {}

    async def _call_tool_records(*a, **kw):
        observed["pending_mcp_jwt"] = server._pending_mcp_jwt
        result = MagicMock()
        result.isError = False
        result.content = []
        result.structuredContent = None
        return result

    session = MagicMock()
    session.call_tool = _call_tool_records
    server.session = session
    server._ready = MagicMock()
    server._ready.is_set.return_value = True

    mcp_tool._servers["srv"] = server
    mcp_tool._server_error_counts.pop("srv", None)
    mcp_tool._ensure_mcp_loop()

    tokens = set_session_vars(mcp_jwt="bridge-test-jwt")
    try:
        handler = _make_tool_handler("srv", "tool1", 10.0)
        handler({"arg": "v"})
    finally:
        clear_session_vars(tokens)
        mcp_tool._servers.pop("srv", None)
        mcp_tool._server_error_counts.pop("srv", None)

    assert observed["pending_mcp_jwt"] == "bridge-test-jwt"
    assert server._pending_mcp_jwt is None  # cleared after the call


def test_make_tool_handler_pending_mcp_jwt_none_when_not_bound(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from unittest.mock import MagicMock

    from tools import mcp_tool
    from tools.mcp_tool import MCPServerTask, _make_tool_handler
    from gateway.session_context import reset_session_vars

    reset_session_vars()

    server = MCPServerTask("srv")
    observed = {}

    async def _call_tool_records(*a, **kw):
        observed["pending_mcp_jwt"] = server._pending_mcp_jwt
        result = MagicMock()
        result.isError = False
        result.content = []
        result.structuredContent = None
        return result

    session = MagicMock()
    session.call_tool = _call_tool_records
    server.session = session
    server._ready = MagicMock()
    server._ready.is_set.return_value = True

    mcp_tool._servers["srv"] = server
    mcp_tool._server_error_counts.pop("srv", None)
    mcp_tool._ensure_mcp_loop()

    try:
        handler = _make_tool_handler("srv", "tool1", 10.0)
        handler({"arg": "v"})
    finally:
        mcp_tool._servers.pop("srv", None)
        mcp_tool._server_error_counts.pop("srv", None)

    assert observed["pending_mcp_jwt"] is None
