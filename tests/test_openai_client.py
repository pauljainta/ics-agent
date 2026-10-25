"""Tests for the stdlib OpenAI-compatible (vLLM) client, against a local stub."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ics_agent.openai_client import OpenAIError, openai_available, openai_chat


class _Stub(BaseHTTPRequestHandler):
    captured: dict = {}
    headers_seen: dict = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {})
        elif self.path == "/v1/models":
            self._json(200, {"data": [{"id": "qwen2.5-32b-instruct"}]})
        else:
            self._json(404, {})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _Stub.captured = json.loads(self.rfile.read(length))
        _Stub.headers_seen = {k.lower(): v for k, v in self.headers.items()}
        if _Stub.captured.get("model") == "boom":
            self._json(500, {"error": "nope"})
        else:
            content = f"echo:{_Stub.captured['messages'][-1]['content'][:10]}"
            self._json(200, {"choices": [{"message": {"role": "assistant",
                                                       "content": content}}]})

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
    base = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    yield base
    httpd.shutdown()


def test_chat_sends_messages_and_returns_content(server):
    out = openai_chat("hello world prompt", model="qwen2.5-32b-instruct",
                      base_url=server, temperature=0.0, max_tokens=256, api_key="tok")
    assert out == "echo:hello worl"
    sent = _Stub.captured
    assert sent["model"] == "qwen2.5-32b-instruct"
    assert sent["messages"][-1] == {"role": "user", "content": "hello world prompt"}
    assert sent["max_tokens"] == 256 and sent["temperature"] == 0.0
    assert _Stub.headers_seen.get("authorization") == "Bearer tok"


def test_system_message_prepended(server):
    openai_chat("task", model="m", base_url=server, system="be careful")
    assert _Stub.captured["messages"][0] == {"role": "system", "content": "be careful"}


def test_chat_raises_on_http_error(server):
    with pytest.raises(OpenAIError):
        openai_chat("x", model="boom", base_url=server)


def test_available(server):
    assert openai_available(server) is True
    assert openai_available("http://127.0.0.1:1/v1", timeout=0.5) is False


def test_build_openai_generate_native_seam(server):
    from ics_agent.plcbench_native import build_openai_generate

    gen = build_openai_generate("qwen2.5-32b-instruct", base_url=server, max_tokens=99)
    assert gen("drive the process") == "echo:drive the "
    assert _Stub.captured["max_tokens"] == 99


def test_probe_and_generator_accept_same_root_or_v1_url(server):
    from ics_agent.plcbench_model_identity import probe_vllm
    from ics_agent.plcbench_native import build_openai_generate

    for base_url in (server, server.removesuffix("/v1")):
        health = probe_vllm(base_url, "qwen2.5-32b-instruct", token="tok")
        assert health["status"] == "healthy"
        gen = build_openai_generate("qwen2.5-32b-instruct", base_url=base_url,
                                    api_key="tok")
        assert gen("drive the process") == "echo:drive the "
        assert _Stub.headers_seen["authorization"] == "Bearer tok"
