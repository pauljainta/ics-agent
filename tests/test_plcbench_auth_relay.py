"""The CHPC relay authenticates every exposed model route."""
from __future__ import annotations

import http.server
import threading
import urllib.error
import urllib.request

from scripts.plcbench_auth_relay import handler_for


def test_relay_restricts_routes_and_checks_key():
    class Upstream(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"data":[]}'
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_args):
            pass

    upstream = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    relay = http.server.ThreadingHTTPServer(
        ('127.0.0.1', 0), handler_for(
            'study-secret', '127.0.0.1',
            f'http://127.0.0.1:{upstream.server_port}'))
    for server in (upstream, relay):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{relay.server_port}'
    try:
        for path, expected in (('/v1/models', 401), ('/invocations', 404)):
            try:
                urllib.request.urlopen(base + path, timeout=2)
            except urllib.error.HTTPError as exc:
                assert exc.code == expected
            else:
                raise AssertionError(f'{path} unexpectedly permitted')
        request = urllib.request.Request(
            base + '/v1/models', headers={'Authorization': 'Bearer study-secret'})
        assert urllib.request.urlopen(request, timeout=2).read() == b'{"data":[]}'
    finally:
        for server in (relay, upstream):
            server.shutdown()
            server.server_close()
