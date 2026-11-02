"""A session chat stream that forwards caller credentials runs on the canonical durable FIFO.

The forwarded X-MCP-Authorization / X-LLM-Authorization / X-Hermes-User-Id headers ride along with
the request (never the admission payload), so the turn is an ordinary admission: its approval is
the authority's generation-fenced prompt and ``POST /v1/runs/{id}/approval`` answers it with the
documented ``{request_id, execution_generation, choice}`` identity.
"""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import socket
import threading

import aiohttp

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon

LLM_JWT = 'caller-llm-jwt'


class ApprovalPeer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        if messages:
            self.server.auth.append(self.headers.get('Authorization'))
        last = messages[-1] if messages else {}
        message = {'role': 'assistant', 'content': 'CONTROL_DONE'}
        if last.get('role') != 'tool' and 'ASK_APPROVAL' in json.dumps(last):
            args = {'command': 'rm -rf ' + shlex.quote(self.server.target.as_posix())}
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'call_control', 'type': 'function', 'function': {'name': 'terminal', 'arguments': json.dumps(args)}}]}
        finish = 'tool_calls' if 'tool_calls' in message else 'stop'
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if body.get('stream') else 'application/json')
        self.end_headers()
        if body.get('stream'):
            delta = dict(message)
            if 'tool_calls' in delta:
                delta['tool_calls'][0]['index'] = 0
            data = {'id': 'control', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
            self.wfile.write(('data: ' + json.dumps(data) + '\n\ndata: [DONE]\n\n').encode())
        else:
            self.wfile.write(json.dumps({'id': 'control', 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}]}).encode())


async def _sse_events(response):
    name = None
    async for raw in response.content:
        line = raw.decode().rstrip('\n')
        if line.startswith('event: '):
            name = line[len('event: '):]
        elif line.startswith('data: '):
            yield name, json.loads(line[len('data: '):])
            name = None


def test_forwarded_session_stream_approval_uses_canonical_identity(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700); user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), ApprovalPeer)
    peer.auth = []
    peer.target = home / 'owned-removal-target'
    peer.target.mkdir()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{peer.server_port}/v1'
    routes = {'gw': {'model': 'local-wire-stub', 'base_url': base, 'forward_caller_jwt': True}}
    cfg = {'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': base},
           'gateway': {'platforms': {'api_server': {'enabled': True, 'port': port, 'host': '127.0.0.1',
                                                    'model_routes': routes}},
                       'multiplex_profiles': False},
           'auxiliary': {'title_generation': {'enabled': False}}, 'terminal': {'cwd': str(home)},
           'platform_toolsets': {'api_server': ['terminal']}, 'approvals': {'mode': 'manual'}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    env = {**child_env(), **{k: os.environ[k] for k in ('TIRITH_ENABLED',) if k in os.environ}}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, API_SERVER_KEY='ordinary-daemon-owned-secret',
               API_SERVER_ENABLED='true', API_SERVER_PORT=str(port), PYTHONUNBUFFERED='1')

    async def exercise():
        auth = {'Authorization': 'Bearer ordinary-daemon-owned-secret'}
        forwarded = {**auth, 'X-MCP-Authorization': 'Bearer caller-mcp-jwt',
                     'X-LLM-Authorization': 'Bearer ' + LLM_JWT, 'X-Hermes-User-Id': 'someone@example.com'}
        api = f'http://127.0.0.1:{port}'
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as client:
            async with client.post(api + '/api/sessions', json={'id': 'fwd-session'}, headers=auth) as created:
                assert created.status in (200, 201), await created.text()
            seen, answered = [], None
            async with client.post(api + '/api/sessions/fwd-session/chat/stream', headers=forwarded,
                                   json={'message': 'ASK_APPROVAL', 'model': 'gw'}) as stream:
                assert stream.status == 200, await stream.text()
                async for name, data in _sse_events(stream):
                    seen.append(name)
                    if name == 'approval.request':
                        assert type(data.get('execution_generation')) is int, data
                        assert peer.target.exists()
                        body = {'request_id': data['request_id'], 'execution_generation': data['execution_generation'],
                                'choice': 'once'}
                        async with client.post(f"{api}/v1/runs/{data['run_id']}/approval", json=body, headers=auth) as resp:
                            answered = (resp.status, await resp.json())
                    if name == 'done':
                        break
            assert answered is not None and answered[0] == 200, (answered, seen)
            assert answered[1]['status'] == 'resolved', answered
            assert 'run.completed' in seen, seen
            assert not peer.target.exists()
            assert peer.auth and all(a == 'Bearer ' + LLM_JWT for a in peer.auth), peer.auth
    try:
        with daemon(root, home, env, barrier=False):
            asyncio.run(asyncio.wait_for(exercise(), 90))
    finally:
        peer.shutdown(); peer.server_close()
