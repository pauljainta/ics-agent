"""Tests for the stdlib Ollama client, against a local HTTP stub."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ics_agent.ollama_client import OllamaError, ollama_available, ollama_generate


class _Stub(BaseHTTPRequestHandler):
    captured: dict = {}

    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        if self.path == "/api/tags":
            self._json(200, {"models": []})
        else:
            self._json(404, {})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length))
        _Stub.captured = req
        if req.get("model") == "boom":
            self._json(500, {"error": "nope"})
        else:
            self._json(200, {"response": f"echo:{req['prompt'][:10]}"})

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield host
    httpd.shutdown()


def test_generate_sends_options_and_returns_text(server):
    out = ollama_generate("hello world prompt", model="llama3.1", host=server,
                          temperature=0.0, num_predict=256)
    assert out == "echo:hello worl"
    sent = _Stub.captured
    assert sent["model"] == "llama3.1"
    assert sent["stream"] is False
    assert sent["options"]["temperature"] == 0.0
    assert sent["options"]["num_predict"] == 256


def test_generate_raises_on_http_error(server):
    with pytest.raises(OllamaError):
        ollama_generate("x", model="boom", host=server)


def test_available_true_and_false(server):
    assert ollama_available(server) is True
    assert ollama_available("http://127.0.0.1:1", timeout=0.5) is False


def test_build_ollama_generate_native_seam(server):
    from ics_agent.plcbench_native import build_ollama_generate

    gen = build_ollama_generate("llama3.1:latest", host=server, num_predict=128)
    assert gen("drive the process") == "echo:drive the "
    assert _Stub.captured["model"] == "llama3.1:latest"
    assert _Stub.captured["options"]["num_predict"] == 128
