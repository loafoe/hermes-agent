"""Forwarded caller JWTs (agent-jwt-forwarding) must not leak into
auxiliary/fallback LLM clients — they are scoped to the primary chat call
only. See docs/superpowers/plans/2026-07-25-llm-jwt-forwarding.md Task 3."""

from __future__ import annotations


def test_set_runtime_main_omits_api_key_when_forwarded_jwt_flag_set():
    from agent.auxiliary_client import _normalize_main_runtime, reset_runtime_main, set_runtime_main

    token = set_runtime_main(
        "custom", "gpt-5",
        base_url="https://agentgateway.internal/v1",
        api_key="",  # caller-forwarding path publishes an empty string, not the JWT
    )
    try:
        normalized = _normalize_main_runtime(None)
        assert normalized.get("api_key", "") == ""
        assert normalized.get("base_url") == "https://agentgateway.internal/v1"
    finally:
        reset_runtime_main(token)


def test_set_runtime_main_still_publishes_static_api_key_normally():
    """Regression guard: the ordinary (non-forwarded) case must be unaffected —
    a static config api_key still flows through to aux clients as before."""
    from agent.auxiliary_client import _normalize_main_runtime, reset_runtime_main, set_runtime_main

    token = set_runtime_main(
        "custom", "gpt-5",
        base_url="https://agentgateway.internal/v1",
        api_key="static-configured-key",
    )
    try:
        normalized = _normalize_main_runtime(None)
        assert normalized.get("api_key") == "static-configured-key"
    finally:
        reset_runtime_main(token)


def test_agent_init_marks_forwarded_jwt_flag_when_route_forwards(monkeypatch):
    """agent._forwarded_caller_jwt_in_use is set True only when this agent's
    api_key came from Task 2's caller-JWT substitution, never for a plain
    static api_key/base_url pair."""
    from run_agent import AIAgent

    agent = AIAgent(
        base_url="https://agentgateway.internal/v1",
        api_key="caller.jwt.value",
        provider="custom",
        model="claude-sonnet-4-6",
        forwarded_caller_jwt=True,
        quiet_mode=True,
    )
    assert agent._forwarded_caller_jwt_in_use is True


def test_agent_init_flag_false_for_ordinary_static_key():
    from run_agent import AIAgent

    agent = AIAgent(
        base_url="https://agentgateway.internal/v1",
        api_key="static-configured-key",
        provider="custom",
        model="claude-sonnet-4-6",
        quiet_mode=True,
    )
    assert agent._forwarded_caller_jwt_in_use is False
