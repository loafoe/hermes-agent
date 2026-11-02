"""Fork: a durable API admission's forwarded caller credentials and caller user id.

``X-MCP-Authorization`` / ``X-LLM-Authorization`` JWTs and ``X-Hermes-User-Id`` are request-owned,
like the admission's observers (``observe_api_turn``): held in memory while a request observes its
admission, never written to admission JSON (the rule upstream applies to route credentials). The
executing turn snapshots them once (``prepare_api_execution``). A turn no request observes any more
(e.g. one recovered after a restart) runs without them, so a JWT-forwarding MCP server or LLM route
refuses it instead of it running as nobody in particular.
"""
from contextlib import contextmanager

FORWARDED_KEYS = ('forwarded_mcp_jwt', 'forwarded_llm_jwt', 'caller_user_id')


def forwarded_kwargs(**values):
    """The forwarded values a caller actually supplied, for ``observe_api_turn(**kwargs)``."""
    return {key: values[key] for key in FORWARDED_KEYS if values.get(key)}


@contextmanager
def hold_forwarded(authority, admission_id, kwargs):
    held = forwarded_kwargs(**kwargs)
    if not held:
        yield
        return
    registry = getattr(authority, 'api_forwarded', None)
    if registry is None:
        registry = authority.api_forwarded = {}
    entries = registry.setdefault(admission_id, [])
    entries.append(held)
    try:
        yield
    finally:
        entries.remove(held)
        if not entries:
            registry.pop(admission_id, None)


def executing_forwarded(authority, session_id):
    """The newest observing request's values for *session_id*'s executing admission (an exact
    retry that joined it carries the fresher JWTs)."""
    execution = authority.sessions[session_id].event_stream.execution
    admission_id = execution.get('admission_id') if execution else None
    entries = getattr(authority, 'api_forwarded', {}).get(admission_id) or ()
    return dict(entries[-1]) if entries else {}


def executing_value(key):
    """*key* of the executing API admission's forwarded snapshot, else None."""
    from gateway.session_api_turn import api_execution
    current = api_execution.get()
    return (current.get('forwarded') or {}).get(key) if current else None
