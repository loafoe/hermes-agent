"""Tests for the forward_jwt fast-fail path (see #forward_jwt 401 hang fix).

A ``forward_jwt`` server's shared MCP transport reconnects on a 401/503 —
correct, since other callers on the same connection may have valid JWTs.
But the *specific* tool call that carried the rejected JWT must not wait
out the full tool timeout: it was awaiting a response on the now-orphaned
old session, which nothing ever resolves. These tests cover the 3 pieces
that make it fail fast instead:

  1. ``_reconnect_or_reraise_group`` records a timestamp on the server
     when it sees a 401/503 for a forward_jwt server (still returns
     "reconnect" — the transport-level behavior for other callers is
     unchanged).
  2. ``_run_on_mcp_loop``'s new ``fail_fast`` callback raises
     ``ForwardedJwtAuthError`` promptly instead of polling to timeout.
  3. ``_make_forward_jwt_fail_fast``'s ``call_started_at`` guard isolates a
     stale failure from a previous call from poisoning a fresh one.
"""
import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest


def _group_with_status(status_code: int) -> BaseExceptionGroup:
    response = MagicMock()
    response.status_code = status_code
    exc = httpx.HTTPStatusError("boom", request=MagicMock(), response=response)
    return BaseExceptionGroup("unhandled errors in a TaskGroup", [exc])


class TestGroupHasForwardJwtAuthFailure:
    def test_detects_401(self):
        from tools.mcp_tool import _group_has_forward_jwt_auth_failure

        assert _group_has_forward_jwt_auth_failure(_group_with_status(401)) is True

    def test_detects_503(self):
        """mt-mcp-grafana deliberately reports auth failures as 503, not 401."""
        from tools.mcp_tool import _group_has_forward_jwt_auth_failure

        assert _group_has_forward_jwt_auth_failure(_group_with_status(503)) is True

    def test_rejects_500(self):
        from tools.mcp_tool import _group_has_forward_jwt_auth_failure

        assert _group_has_forward_jwt_auth_failure(_group_with_status(500)) is False

    def test_rejects_non_http_exception(self):
        from tools.mcp_tool import _group_has_forward_jwt_auth_failure

        eg = BaseExceptionGroup("boom", [RuntimeError("unrelated")])
        assert _group_has_forward_jwt_auth_failure(eg) is False


class TestReconnectOrReraiseGroupRecordsFailure:
    def _make_server(self, auth_type: str):
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("srv")
        server._auth_type = auth_type
        server._ready.set()
        return server

    def test_forward_jwt_401_records_timestamp_and_still_reconnects(self):
        server = self._make_server("forward_jwt")
        assert server._last_forward_jwt_auth_failure_at is None

        before = time.monotonic()
        result = server._reconnect_or_reraise_group(_group_with_status(401))
        after = time.monotonic()

        assert result == "reconnect"
        assert server._last_forward_jwt_auth_failure_at is not None
        assert before <= server._last_forward_jwt_auth_failure_at <= after

    def test_forward_jwt_503_records_timestamp(self):
        server = self._make_server("forward_jwt")
        server._reconnect_or_reraise_group(_group_with_status(503))
        assert server._last_forward_jwt_auth_failure_at is not None

    def test_non_forward_jwt_server_unaffected(self):
        """oauth/no-auth servers keep today's behavior — no timestamp, just reconnect."""
        server = self._make_server("oauth")
        result = server._reconnect_or_reraise_group(_group_with_status(401))
        assert result == "reconnect"
        assert server._last_forward_jwt_auth_failure_at is None

    def test_forward_jwt_non_auth_failure_does_not_record(self):
        """A transient 500/timeout on a forward_jwt server isn't an auth failure."""
        server = self._make_server("forward_jwt")
        server._reconnect_or_reraise_group(_group_with_status(500))
        assert server._last_forward_jwt_auth_failure_at is None


class TestRunOnMcpLoopFailFast:
    def test_fail_fast_raises_promptly(self):
        from tools.mcp_tool import _run_on_mcp_loop, ForwardedJwtAuthError, _ensure_mcp_loop

        _ensure_mcp_loop()

        async def _slow():
            await asyncio.sleep(30)
            return "should not get here"

        start = time.monotonic()
        with pytest.raises(ForwardedJwtAuthError, match="rejected"):
            _run_on_mcp_loop(_slow, timeout=30, fail_fast=lambda: "rejected the forwarded caller JWT")
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, f"fail_fast should short-circuit almost immediately, took {elapsed:.2f}s"

    def test_no_fail_fast_behaves_as_before(self):
        """fail_fast returning None on every tick must not change normal completion."""
        from tools.mcp_tool import _run_on_mcp_loop, _ensure_mcp_loop

        _ensure_mcp_loop()

        async def _quick():
            await asyncio.sleep(0.05)
            return "done"

        result = _run_on_mcp_loop(_quick, timeout=5, fail_fast=lambda: None)
        assert result == "done"

    def test_omitted_fail_fast_behaves_as_before(self):
        from tools.mcp_tool import _run_on_mcp_loop, _ensure_mcp_loop

        _ensure_mcp_loop()

        async def _quick():
            return "done"

        assert _run_on_mcp_loop(_quick, timeout=5) == "done"


class TestMakeForwardJwtFailFast:
    def test_returns_none_for_non_forward_jwt_server(self):
        from tools.mcp_tool import _make_forward_jwt_fail_fast

        server = MagicMock()
        server._auth_type = "oauth"
        assert _make_forward_jwt_fail_fast(server, "srv", time.monotonic()) is None

    def test_fresh_failure_after_call_start_triggers(self):
        from tools.mcp_tool import _make_forward_jwt_fail_fast

        server = MagicMock()
        server._auth_type = "forward_jwt"
        call_started_at = time.monotonic()
        server._last_forward_jwt_auth_failure_at = call_started_at + 0.01

        fail_fast = _make_forward_jwt_fail_fast(server, "srv", call_started_at)
        message = fail_fast()
        assert message is not None
        assert "srv" in message
        assert "rejected the forwarded caller JWT" in message

    def test_stale_failure_before_call_start_does_not_trigger(self):
        """A previous, unrelated call's bad JWT must not poison a fresh call."""
        from tools.mcp_tool import _make_forward_jwt_fail_fast

        server = MagicMock()
        server._auth_type = "forward_jwt"
        call_started_at = time.monotonic()
        server._last_forward_jwt_auth_failure_at = call_started_at - 5.0

        fail_fast = _make_forward_jwt_fail_fast(server, "srv", call_started_at)
        assert fail_fast() is None

    def test_no_failure_recorded_does_not_trigger(self):
        from tools.mcp_tool import _make_forward_jwt_fail_fast

        server = MagicMock()
        server._auth_type = "forward_jwt"
        server._last_forward_jwt_auth_failure_at = None

        fail_fast = _make_forward_jwt_fail_fast(server, "srv", time.monotonic())
        assert fail_fast() is None


def test_call_tool_handler_fast_fails_on_forward_jwt_rejection(monkeypatch, tmp_path):
    """End-to-end through _make_tool_handler: a forward_jwt server whose shared
    transport already recorded an auth failure at/after this call's start
    must return needs_reauth immediately, not hang toward the tool timeout."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools.mcp_tool import _make_tool_handler
    from tools import mcp_tool

    server = MagicMock()
    server.name = "srv"
    server._auth_type = "forward_jwt"
    session = MagicMock()

    async def _call_tool_hangs(*a, **kw):
        # Simulates the real bug: the call is blocked on an orphaned response
        # stream that never resolves on its own within the test's patience.
        await asyncio.sleep(30)
        return SimpleNamespace(isError=False, content=[], structuredContent=None)

    session.call_tool = _call_tool_hangs
    server.session = session
    server._rpc_lock = asyncio.Lock()
    server.mark_tool_call = MagicMock()

    mcp_tool._servers["srv"] = server
    mcp_tool._server_error_counts.pop("srv", None)
    mcp_tool._ensure_mcp_loop()

    # Set the failure timestamp slightly in the future relative to "now" so
    # it's unambiguously >= whatever call_started_at the handler captures.
    server._last_forward_jwt_auth_failure_at = time.monotonic() + 0.05

    try:
        handler = _make_tool_handler("srv", "tool1", 30.0)
        start = time.monotonic()
        result = handler({"arg": "v"})
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"expected fast-fail, took {elapsed:.2f}s"
        parsed = json.loads(result)
        assert parsed.get("needs_reauth") is True
        assert parsed.get("server") == "srv"
        assert "rejected the forwarded caller JWT" in parsed.get("error", "")
    finally:
        mcp_tool._servers.pop("srv", None)
        mcp_tool._server_error_counts.pop("srv", None)


def test_call_tool_handler_forward_jwt_server_without_failure_uses_normal_path(monkeypatch, tmp_path):
    """A forward_jwt server with no recorded failure behaves exactly like
    before this fix — success path unaffected."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools.mcp_tool import _make_tool_handler
    from tools import mcp_tool

    server = MagicMock()
    server.name = "srv"
    server._auth_type = "forward_jwt"
    server._last_forward_jwt_auth_failure_at = None
    session = MagicMock()

    async def _call_tool_ok(*a, **kw):
        return SimpleNamespace(isError=False, content=[], structuredContent=None)

    session.call_tool = _call_tool_ok
    server.session = session
    server._rpc_lock = asyncio.Lock()
    server.mark_tool_call = MagicMock()

    mcp_tool._servers["srv"] = server
    mcp_tool._server_error_counts.pop("srv", None)
    mcp_tool._ensure_mcp_loop()

    try:
        handler = _make_tool_handler("srv", "tool1", 10.0)
        result = handler({"arg": "v"})
        parsed = json.loads(result)
        assert "needs_reauth" not in parsed
        assert parsed.get("result") == ""
    finally:
        mcp_tool._servers.pop("srv", None)
        mcp_tool._server_error_counts.pop("srv", None)
