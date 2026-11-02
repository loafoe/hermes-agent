"""Fork: forwarded caller credentials reach a durable API turn request-owned, never its payload."""
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

FORWARDED = {'X-MCP-Authorization': 'Bearer caller-mcp-jwt', 'X-LLM-Authorization': 'Bearer caller-llm-jwt',
             'X-Hermes-User-Id': 'someone@example.com'}
VALUES = ('caller-mcp-jwt', 'caller-llm-jwt', 'someone@example.com')


def _probe_turn(owner, seen):
    from gateway.run import GatewayRunner
    from gateway.session import SessionContext
    from gateway.session_api_forwarding import executing_value
    from gateway.session_context import clear_session_vars, get_session_mcp_jwt

    async def handle(event):
        from gateway.session_results import execution_result
        context = SessionContext(source=event.source, connected_platforms=[], home_channels={},
                                 session_key='k', session_id=event.source.chat_id)
        tokens = GatewayRunner._set_session_env(owner.runner, context)
        try:
            seen.append({'mcp': get_session_mcp_jwt(), 'llm': executing_value('forwarded_llm_jwt'),
                         'user': executing_value('caller_user_id')})
        finally:
            clear_session_vars(tokens)
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'
    owner.runner._handle_message = handle


async def _stream(api, headers):
    app = web.Application()
    app.router.add_post('/api/sessions/{session_id}/chat/stream', api._handle_session_chat_stream)
    async with TestClient(TestServer(app)) as client:
        response = await client.post('/api/sessions/fwd/chat/stream', json={'message': 'hi'},
                                     headers={'Authorization': 'Bearer forwarded-values-key', **headers})
        return await response.text()


@pytest.mark.asyncio
async def test_session_stream_turn_runs_with_the_requests_forwarded_values(api, owner):
    seen = []
    _probe_turn(owner, seen)
    api._api_key = 'forwarded-values-key'
    owner.db.create_session('fwd', source='api_server')
    text = await _stream(api, FORWARDED)
    assert 'event: run.completed' in text, text
    assert seen == [{'mcp': 'caller-mcp-jwt', 'llm': 'caller-llm-jwt', 'user': 'someone@example.com'}]
    assert not getattr(owner, 'api_forwarded', {})  # released with the request
    with owner.db._read_ctx() as conn:
        stored = json.dumps([tuple(row) for row in conn.execute('SELECT * FROM session_admissions')], default=str)
    assert stored != '[]'
    assert not [value for value in VALUES if value in stored]


@pytest.mark.asyncio
async def test_turn_without_forwarded_headers_binds_none(api, owner):
    seen = []
    _probe_turn(owner, seen)
    api._api_key = 'forwarded-values-key'
    owner.db.create_session('fwd', source='api_server')
    assert 'event: run.completed' in await _stream(api, {})
    assert seen == [{'mcp': None, 'llm': None, 'user': None}]


@pytest.mark.asyncio
@pytest.mark.parametrize('path, body, extra', [
    ('/v1/runs', {'input': 'hi'}, {}),
    ('/v1/responses', {'input': 'hi'}, {'Idempotency-Key': 'fwd-1'}),
])
async def test_other_durable_doors_forward_their_jwts(api, owner, path, body, extra):
    from tests.waiting import async_wait_until
    seen = []
    _probe_turn(owner, seen)
    api._api_key = 'forwarded-values-key'
    app = web.Application()
    app.router.add_post('/v1/runs', api._handle_runs)
    app.router.add_post('/v1/responses', api._handle_responses)
    headers = {'Authorization': 'Bearer forwarded-values-key', **FORWARDED, **extra}
    async with TestClient(TestServer(app)) as client:
        response = await client.post(path, json=body, headers=headers)
        assert response.status in (200, 202), await response.text()
        await async_wait_until(lambda: seen, what='the admitted turn to execute')
    assert seen[0]['mcp'] == 'caller-mcp-jwt' and seen[0]['llm'] == 'caller-llm-jwt'


def test_newest_observing_request_wins_and_each_releases_its_own():
    from types import SimpleNamespace
    from gateway.session_api_forwarding import executing_forwarded, hold_forwarded
    authority = SimpleNamespace(sessions={'s': SimpleNamespace(event_stream=SimpleNamespace(
        execution={'admission_id': 'a1'}))})
    with hold_forwarded(authority, 'a1', {'forwarded_mcp_jwt': 'old'}):
        with hold_forwarded(authority, 'a1', {'forwarded_mcp_jwt': 'new', 'turn_author': 'ignored'}):
            assert executing_forwarded(authority, 's') == {'forwarded_mcp_jwt': 'new'}
        assert executing_forwarded(authority, 's') == {'forwarded_mcp_jwt': 'old'}
    assert executing_forwarded(authority, 's') == {} and authority.api_forwarded == {}
