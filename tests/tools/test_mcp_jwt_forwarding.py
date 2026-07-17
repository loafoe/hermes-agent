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
