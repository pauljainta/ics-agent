"""The CHPC D3 relay exposes only authenticated System One traffic."""
from __future__ import annotations

import http.server
import threading
import urllib.error
import urllib.request

from scripts.plcbench_systemone_relay import handler_for


def test_systemone_relay_checks_source_key_and_routes():
    class Upstream(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Length', '2')
            self.end_headers()
            self.wfile.write(b'{}')

        def do_POST(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    upstream = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    relay = http.server.ThreadingHTTPServer(
        ('127.0.0.1', 0), handler_for(
            'd3-secret', '127.0.0.1',
            f'http://127.0.0.1:{upstream.server_port}'))
    blocked_source = http.server.ThreadingHTTPServer(
        ('127.0.0.1', 0), handler_for(
            'd3-secret', '192.0.2.1',
            f'http://127.0.0.1:{upstream.server_port}'))
    for server in (upstream, relay, blocked_source):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{relay.server_port}'
    try:
        for path, expected in (('/health', 401), ('/v1/models', 401),
                               ('/v1/chat/completions', 404)):
            try:
                urllib.request.urlopen(base + path, timeout=2)
            except urllib.error.HTTPError as exc:
                assert exc.code == expected
            else:
                raise AssertionError(f'{path} unexpectedly permitted')
        request = urllib.request.Request(
            base + '/v1/systemone', data=b'{"questions":{}}',
            headers={'Authorization': 'Bearer d3-secret',
                     'Content-Type': 'application/json'})
        assert urllib.request.urlopen(request, timeout=2).read() == b'{"questions":{}}'
        denied = urllib.request.Request(
            f'http://127.0.0.1:{blocked_source.server_port}/v1/models',
            headers={'Authorization': 'Bearer d3-secret'})
        try:
            urllib.request.urlopen(denied, timeout=2)
        except urllib.error.HTTPError as exc:
            assert exc.code == 403
        else:
            raise AssertionError('other source IP unexpectedly permitted')
    finally:
        for server in (relay, blocked_source, upstream):
            server.shutdown()
            server.server_close()
