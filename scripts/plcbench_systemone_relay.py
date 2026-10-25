#!/usr/bin/env python3
"""CHPC login-only, bearer-authenticated relay to compute-loopback D3.

The relay accepts only the System One routes and logs no bodies or credentials.
"""
from __future__ import annotations

import argparse
import hmac
import http.server
import ipaddress
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

ROUTES = {('GET', '/health'), ('GET', '/v1/models'),
          ('POST', '/v1/systemone')}
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


def handler_for(key: str, allowed_ip: str, upstream: str):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def _reply(self, status: int, body: bytes = b'', content_type: str = 'text/plain'):
            print(f'{self.client_address[0]} {self.command} '
                  f'{urlsplit(self.path).path} {status}', file=sys.stderr, flush=True)
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def _handle(self):
            path = urlsplit(self.path)
            if (path.scheme or path.netloc or path.query or path.fragment
                    or (self.command, path.path) not in ROUTES):
                return self._reply(404)
            if self.client_address[0] != allowed_ip:
                return self._reply(403)
            auth = self.headers.get('Authorization', '')
            if not hmac.compare_digest(auth, 'Bearer ' + key):
                return self._reply(401)
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                return self._reply(400)
            if length < 0 or length > MAX_REQUEST_BYTES:
                return self._reply(413)
            if self.command == 'POST' and not length:
                return self._reply(400)
            body = self.rfile.read(length) if length else None
            headers = {'Authorization': auth}
            if body is not None:
                headers['Content-Type'] = 'application/json'
            request = urllib.request.Request(
                upstream + path.path, data=body, headers=headers,
                method=self.command)
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    content = response.read(MAX_RESPONSE_BYTES + 1)
                    if len(content) > MAX_RESPONSE_BYTES:
                        return self._reply(502, b'upstream response too large')
                    self._reply(response.status, content,
                                response.headers.get('Content-Type', 'application/json'))
            except urllib.error.HTTPError as exc:
                self._reply(exc.code, exc.read(MAX_RESPONSE_BYTES),
                            exc.headers.get('Content-Type', 'application/json'))
            except (OSError, TimeoutError):
                self._reply(502, b'upstream unavailable')

        def do_GET(self):
            self._handle()

        def do_POST(self):
            self._handle()

        def log_message(self, format, *args):
            # Never log the request line, which might contain a query string.
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-file', required=True, type=Path)
    parser.add_argument('--allow-ip', required=True)
    parser.add_argument('--listen-host', required=True)
    parser.add_argument('--listen-port', type=int, default=18081)
    args = parser.parse_args()
    ipaddress.ip_address(args.allow_ip)
    bind = ipaddress.ip_address(args.listen_host)
    if bind.is_unspecified or bind.is_loopback:
        raise ValueError('relay must bind a specific compute interface')
    key = args.key_file.read_text().strip()
    if not key:
        raise RuntimeError('empty relay key')
    server = http.server.ThreadingHTTPServer(
        (str(bind), args.listen_port),
        handler_for(key, args.allow_ip, 'http://127.0.0.1:18080'))
    server.serve_forever()


if __name__ == '__main__':
    main()
