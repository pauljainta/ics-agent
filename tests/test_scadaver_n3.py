from __future__ import annotations

import json
import subprocess
from urllib.error import HTTPError
from pathlib import Path

from experiments.scadaver_sphere.n3 import (
    EventJournal,
    GET_SKILL_SCHEMA,
    N3Executor,
    SKILL_NAMES,
    _runtime_tool_manifest,
    tool_schemas,
)
from experiments.scadaver_sphere.n3_agent import run_n3_agent
from experiments.scadaver_sphere.n3_inference import N3OpenRouterClient
from experiments.scadaver_sphere import n3_inference
from experiments.scadaver_sphere.n3_transport import N3Transport
from experiments.scadaver_sphere.budgets import EpisodeBudget, EpisodeLimits
from experiments.scadaver_sphere.integrated_runner import validate_report
from experiments.scadaver_sphere.policy import ReadOnlyPolicy
from experiments.scadaver_sphere.n3_worker import N3ReadOnlyWorker, N3_WORKER_TOOL_SCHEMAS


def _runner(argv: list[str], _timeout: float) -> subprocess.CompletedProcess[str]:
    if argv[:4] == ["ip", "-j", "address", "show"]:
        return subprocess.CompletedProcess(argv, 0, json.dumps([{
            "ifname": "eth1", "addr_info": [{
                "family": "inet", "local": "198.18.4.30", "prefixlen": 24,
                "scope": "global",
            }],
        }]), "")
    if argv[:4] == ["ip", "-j", "route", "show"]:
        return subprocess.CompletedProcess(argv, 0, json.dumps([{
            "dst": "198.18.4.0/24", "dev": "eth1", "scope": "link",
        }]), "")
    raise AssertionError(argv)


def test_n3_foothold_context_is_observed_without_role_leakage() -> None:
    worker = N3ReadOnlyWorker(
        ReadOnlyPolicy.build("198.18.4.0/24", (1502, 1503), max_hosts=8),
        runner=_runner, live=True,
    )
    result = worker.execute("get_foothold_context", {})
    assert result["status"] == "ok"
    assert result["host_role"] == "initial_access_vm"
    assert result["interfaces"][0]["ipv4"][0]["address"] == "198.18.4.30"
    assert "controller" not in json.dumps(result).lower()
    assert "simulator" not in json.dumps(result).lower()


def test_n3_service_enumeration_returns_only_observed_open_ports(monkeypatch) -> None:
    worker = N3ReadOnlyWorker(
        ReadOnlyPolicy.build("198.18.4.0/24", (1502, 1503), max_hosts=8), live=True,
    )
    monkeypatch.setattr(worker, "_tcp_open", lambda _ip, port: port == 1503)
    result = worker.execute("enumerate_services", {"ip": "198.18.4.20"})
    assert result["status"] == "ok"
    assert result["services"] == [{"port": 1503, "transport": "tcp", "reachable": True}]
    assert result["candidate_count"] == 2


def test_n3_tool_surface_is_closed_and_has_no_write_primitive() -> None:
    names = [item["function"]["name"] for item in N3_WORKER_TOOL_SCHEMAS]
    assert {"get_foothold_context", "enumerate_services"} <= set(names)
    assert not any("write" in name or "shell" in name for name in names)
    schemas = tool_schemas(True, 1.25)
    assert GET_SKILL_SCHEMA in schemas
    assert [item["function"]["name"] for item in schemas].count("bounded_binary_output") == 1
    assert all(item["function"]["parameters"]["additionalProperties"] is False for item in schemas)


def test_n3_transport_requests_internal_session_action_authorization(monkeypatch) -> None:
    transport = N3Transport("http://worker.invalid")
    requests: list[tuple[str, dict]] = []

    def request_fixture(path: str, body: dict, **_kwargs):
        requests.append((path, body))
        return {
            "session_id": "session-1", "harness_profile": "heterogeneous_n3",
            "tool_names": [item["function"]["name"] for item in N3_WORKER_TOOL_SCHEMAS],
            "action_broker_enabled": True,
        }

    monkeypatch.setattr(transport, "_request", request_fixture)
    result = transport.start_episode(
        {item["function"]["name"] for item in N3_WORKER_TOOL_SCHEMAS},
        autonomous_actions=True,
    )
    assert result["action_broker_enabled"] is True
    assert requests == [("/session", {"autonomous_actions": True})]


def test_n3_executor_uses_internal_session_broker_without_a_user_token(tmp_path: Path) -> None:
    class Transport:
        def __init__(self) -> None:
            self.arguments: dict | None = None

        def execute_action(self, arguments):
            self.arguments = arguments
            return {"tool": "bounded_binary_output", "status": "ok"}

    transport = Transport()
    executor = N3Executor(transport, EventJournal(tmp_path), True)  # type: ignore[arg-type]
    result = executor.execute("bounded_binary_output", {"active": True, "duration_seconds": 1.0})
    assert result["status"] == "ok"
    assert transport.arguments == {"active": True, "duration_seconds": 1.0}


def test_n3_invalid_submission_is_an_agent_visible_schema_rejection(tmp_path: Path) -> None:
    class Transport:
        pass

    executor = N3Executor(Transport(), EventJournal(tmp_path), False)  # type: ignore[arg-type]
    result = executor.execute("submit_integrated_report", {"report": "{\"run_id\":\"double-encoded\"}"})
    assert result == {
        "tool": "submit_integrated_report",
        "status": "rejected",
        "error": "invalid_report_schema",
        "detail": "integrated report must be an object",
    }


def test_n3_runtime_tool_manifest_repeats_only_registered_capabilities() -> None:
    manifest = _runtime_tool_manifest(tool_schemas(True, 1.25))
    assert "bounded_binary_output" in manifest
    assert "submit_integrated_report" in manifest
    assert "10.100.0." not in manifest


def test_n3_local_client_serializes_required_tool_recovery(monkeypatch) -> None:
    body = {
        "id": "n3-response", "model": "qwen/qwen3.8-27b", "provider": "fixture",
        "choices": [{"finish_reason": "tool_calls", "message": {
            "content": None, "tool_calls": [{"id": "tool-1", "type": "function",
                "function": {"name": "get_skill", "arguments": '{"skill_name":"ot-fundamentals"}'}}],
        }}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4},
    }

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(body).encode()

    requests: list[dict] = []

    def open_fixture(request, timeout):
        assert timeout == 240.0
        requests.append(json.loads(request.data))
        return Response()

    monkeypatch.setenv("TEST_N3_KEY", "secret-not-for-records")
    monkeypatch.setattr(n3_inference, "urlopen", open_fixture)
    client = N3OpenRouterClient("qwen/qwen3.8-27b", "TEST_N3_KEY", seed=101)
    response = client.chat([], [], max_tokens=256, call_index=2,
                           tool_choice="required")
    assert response["tool_calls"][0]["function"]["name"] == "get_skill"
    assert requests[0]["tool_choice"] == "required"
    assert client.request_records[-1]["requested_tool_choice"] == "required"
    assert "secret-not-for-records" not in json.dumps(client.request_records)


def test_n3_client_falls_back_only_after_provider_rejects_output_limit(monkeypatch) -> None:
    body = {
        "id": "n3-fallback", "model": "qwen/qwen3.8-27b", "provider": "fixture",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 2},
    }

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(body).encode()

    requested_limits: list[int] = []

    def open_fixture(request, *, timeout):
        assert timeout == 240.0
        requested_limits.append(json.loads(request.data)["max_tokens"])
        if len(requested_limits) == 1:
            raise HTTPError(request.full_url, 400, "unsupported output limit", None, None)
        return Response()

    monkeypatch.setenv("TEST_N3_KEY", "secret-not-for-records")
    monkeypatch.setattr(n3_inference, "urlopen", open_fixture)
    client = N3OpenRouterClient(
        "qwen/qwen3.8-27b", "TEST_N3_KEY", output_token_fallbacks=(8192, 4096))
    response = client.chat([], [], max_tokens=16384)
    assert response["_scadaver_response_metadata"]["requested_max_tokens"] == 8192
    assert requested_limits == [16384, 8192]
    assert client.request_records[0]["output_limit_fallback_to"] == 8192
    assert client.request_records[1]["output_limit_fallback_from"] == 16384


def test_n3_skill_loader_is_allowlisted_and_journaled(tmp_path: Path) -> None:
    class Transport:
        pass

    journal = EventJournal(tmp_path)
    executor = N3Executor(Transport(), journal, None)  # type: ignore[arg-type]
    result = executor.execute("get_skill", {"skill_name": SKILL_NAMES[0]})
    assert result["status"] == "ok"
    assert result["skill_name"] == SKILL_NAMES[0]
    assert "OT Fundamentals" in result["content"]
    assert executor.execute("get_skill", {"skill_name": "private-ground-truth"})["status"] == "rejected"


def test_n3_prompt_files_do_not_contain_private_scope_hints() -> None:
    root = Path(__file__).resolve().parents[1] / "experiments/scadaver_sphere/prompts"
    content = (root / "n3_system.md").read_text() + (root / "n3_mission.md").read_text()
    assert "198.18.4.0/24" not in content
    assert "1502" not in content


def test_n3_production_path_mock_retrieves_skill_and_submits(tmp_path: Path) -> None:
    report = {
        "run_id": "n3-mock", "objective": "fixture", "systems": [],
        "relationships": [], "permission_assessment": [],
        "objective_plan": {"summary": "insufficient evidence", "steps": [],
                           "supporting_observation_ids": [], "requires_action": False,
                           "limitations": ["fixture"]},
        "action_attempted": False, "outcome": "not attempted",
        "outcome_observation_ids": [], "overall_assessment": "fixture complete",
    }

    class Client:
        max_tokens = 512
        request_records = []
        calls = 0

        def chat(self, _messages, _tools, **kwargs):
            self.calls += 1
            self.request_records.append({"usage": {"prompt_tokens": 10, "completion_tokens": 10}})
            if self.calls == 1:
                return {"tool_calls": [{"id": "skill", "function": {
                    "name": "get_skill", "arguments": '{"skill_name":"ot-fundamentals"}'}}],
                        "_scadaver_response_metadata": {"finish_reason": "tool_calls"}}
            return {"tool_calls": [{"id": "submit", "function": {
                "name": "submit_integrated_report",
                "arguments": json.dumps({"report": report})}}],
                    "_scadaver_response_metadata": {"finish_reason": "tool_calls"}}

        def mark_termination(self, _reason):
            return None

    class Transport:
        def execute(self, _name, _arguments):
            raise AssertionError("remote tool not expected")

    journal = EventJournal(tmp_path)
    executor = N3Executor(Transport(), journal, None)  # type: ignore[arg-type]
    outcome = run_n3_agent(
        Client(), executor, max_steps=4, system_prompt="system", mission_prompt="mission",
        initial_context="foothold", tool_schemas=tool_schemas(False, 1.25),
        report_validator=validate_report, final_submission_max_tokens=1024,
        reserved_completion_steps=1, episode_deadline_seconds=30,
        budget=EpisodeBudget(EpisodeLimits(max_model_calls=4, max_tool_calls=4,
                                           max_input_tokens=1000000, max_output_tokens=10000)),
        max_context_chars=100000, context_tail_groups=10, event_sink=journal.record,
        decision_recovery_max_output_tokens=256,
        enable_investigation_truncation_recovery=False,
        force_final_submission_tool_choice=False,
    )
    assert outcome.stop_reason == "submitted_integrated_report"
    assert outcome.report == report
    assert (tmp_path / "events.jsonl").is_file()
    assert (tmp_path / "skill_loads.jsonl").is_file()


def test_n3_finalization_can_correct_a_rejected_report_schema(tmp_path: Path) -> None:
    report = {
        "run_id": "n3-schema-retry", "objective": "fixture", "systems": [],
        "relationships": [], "permission_assessment": [],
        "objective_plan": {"summary": "insufficient evidence", "steps": [],
                           "supporting_observation_ids": [], "requires_action": False,
                           "limitations": ["fixture"]},
        "action_attempted": False, "outcome": "not attempted",
        "outcome_observation_ids": [], "overall_assessment": "fixture complete",
    }

    class Client:
        max_tokens = 512
        request_records: list[dict] = []

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def chat(self, _messages, tools, **kwargs):
            self.calls.append({**kwargs, "tools": tools})
            if len(self.calls) == 1:
                arguments = {"report": json.dumps(report)}
            else:
                arguments = {"report": report}
            return {"tool_calls": [{"id": f"submit-{len(self.calls)}", "function": {
                "name": "submit_integrated_report", "arguments": json.dumps(arguments)}}],
                "_scadaver_response_metadata": {"finish_reason": "tool_calls"}}

        def mark_termination(self, _reason):
            return None

    class Transport:
        pass

    client = Client()
    journal = EventJournal(tmp_path)
    outcome = run_n3_agent(
        client, N3Executor(Transport(), journal, False), max_steps=2,
        system_prompt="system", mission_prompt="mission", initial_context="foothold",
        tool_schemas=tool_schemas(False, 1.25), report_validator=validate_report,
        final_submission_max_tokens=1024, reserved_completion_steps=2,
        episode_deadline_seconds=30,
        budget=EpisodeBudget(EpisodeLimits(max_model_calls=2, max_tool_calls=2,
                                           max_input_tokens=1_000_000, max_output_tokens=10_000)),
        max_context_chars=100_000, context_tail_groups=10,
        decision_recovery_max_output_tokens=256, event_sink=journal.record,
        enable_investigation_truncation_recovery=False,
        force_final_submission_tool_choice=True,
    )
    assert outcome.stop_reason == "submitted_integrated_report"
    assert outcome.report == report
    rows = [json.loads(line) for line in (tmp_path / "observations.jsonl").read_text().splitlines()]
    assert rows[0]["result"]["error"] == "invalid_report_schema"
    assert len(client.calls) == 2
    assert client.calls[1]["max_tokens"] == 1024
    assert client.calls[1]["tool_choice"] == {
        "type": "function", "function": {"name": "submit_integrated_report"},
    }


def test_n3_finalization_recovers_once_from_length_truncation(tmp_path: Path) -> None:
    report = {
        "run_id": "n3-final-length", "objective": "fixture", "systems": [],
        "relationships": [], "permission_assessment": [],
        "objective_plan": {"summary": "insufficient evidence", "steps": [],
                           "supporting_observation_ids": [], "requires_action": False,
                           "limitations": ["fixture"]},
        "action_attempted": False, "outcome": "not attempted",
        "outcome_observation_ids": [], "overall_assessment": "fixture complete",
    }

    class Client:
        max_tokens = 512
        request_records: list[dict] = []

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def chat(self, _messages, tools, **kwargs):
            self.calls.append({**kwargs, "tools": tools})
            if len(self.calls) == 1:
                return {"content": "partial report", "_scadaver_response_metadata": {
                    "finish_reason": "length"}}
            return {"tool_calls": [{"id": "submit", "function": {
                "name": "submit_integrated_report",
                "arguments": json.dumps({"report": report})}}],
                "_scadaver_response_metadata": {"finish_reason": "tool_calls"}}

        def mark_termination(self, _reason):
            return None

    class Transport:
        pass

    client = Client()
    outcome = run_n3_agent(
        client, N3Executor(Transport(), EventJournal(tmp_path), False), max_steps=2,
        system_prompt="system", mission_prompt="mission", initial_context="foothold",
        tool_schemas=tool_schemas(False, 1.25), report_validator=validate_report,
        final_submission_max_tokens=2048, reserved_completion_steps=2,
        episode_deadline_seconds=30,
        budget=EpisodeBudget(EpisodeLimits(max_model_calls=2, max_tool_calls=2,
                                           max_input_tokens=1_000_000, max_output_tokens=10_000)),
        max_context_chars=100_000, context_tail_groups=10,
        decision_recovery_max_output_tokens=128, event_sink=lambda *_args: None,
        enable_investigation_truncation_recovery=True,
        force_final_submission_tool_choice=True,
    )
    assert outcome.stop_reason == "submitted_integrated_report"
    assert [call["max_tokens"] for call in client.calls] == [2048, 2048]
    assert all(call["tool_choice"] == {
        "type": "function", "function": {"name": "submit_integrated_report"},
    } for call in client.calls)


def test_n3_second_invalid_submission_terminates_accurately(tmp_path: Path) -> None:
    class Client:
        max_tokens = 512
        request_records: list[dict] = []

        def __init__(self) -> None:
            self.calls = 0

        def chat(self, _messages, _tools, **_kwargs):
            self.calls += 1
            return {"tool_calls": [{"id": f"bad-{self.calls}", "function": {
                "name": "submit_integrated_report",
                "arguments": json.dumps({"report": "still-not-an-object"})}}],
                "_scadaver_response_metadata": {"finish_reason": "tool_calls"}}

        def mark_termination(self, _reason):
            return None

    class Transport:
        pass

    client = Client()
    outcome = run_n3_agent(
        client, N3Executor(Transport(), EventJournal(tmp_path), False), max_steps=3,
        system_prompt="system", mission_prompt="mission", initial_context="foothold",
        tool_schemas=tool_schemas(False, 1.25), report_validator=validate_report,
        final_submission_max_tokens=2048, reserved_completion_steps=3,
        episode_deadline_seconds=30,
        budget=EpisodeBudget(EpisodeLimits(max_model_calls=3, max_tool_calls=3,
                                           max_input_tokens=1_000_000, max_output_tokens=10_000)),
        max_context_chars=100_000, context_tail_groups=10,
        decision_recovery_max_output_tokens=128, event_sink=lambda *_args: None,
        enable_investigation_truncation_recovery=True,
        force_final_submission_tool_choice=True,
    )
    assert outcome.stop_reason == "invalid_final_report"
    assert client.calls == 2


def test_n3_truncation_uses_one_required_tool_recovery_then_continues(tmp_path: Path) -> None:
    report = {
        "run_id": "n3-recovery", "objective": "fixture", "systems": [],
        "relationships": [], "permission_assessment": [],
        "objective_plan": {"summary": "insufficient evidence", "steps": [],
                           "supporting_observation_ids": [], "requires_action": False,
                           "limitations": ["fixture"]},
        "action_attempted": False, "outcome": "not attempted",
        "outcome_observation_ids": [], "overall_assessment": "fixture complete",
    }

    class Client:
        max_tokens = 4096
        request_records: list[dict] = []

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def chat(self, _messages, tools, **kwargs):
            self.calls.append({**kwargs, "tools": tools})
            if len(self.calls) == 1:
                return {"content": "partial", "_scadaver_response_metadata": {"finish_reason": "length"}}
            if len(self.calls) == 2:
                return {"tool_calls": [{"id": "recover", "function": {
                    "name": "get_skill", "arguments": '{"skill_name":"ot-fundamentals"}'}}],
                    "_scadaver_response_metadata": {"finish_reason": "tool_calls"}}
            return {"tool_calls": [{"id": "submit", "function": {
                "name": "submit_integrated_report", "arguments": json.dumps({"report": report})}}],
                    "_scadaver_response_metadata": {"finish_reason": "tool_calls"}}

        def mark_termination(self, _reason):
            return None

    class Transport:
        def execute(self, _name, _arguments):
            raise AssertionError("remote tool not expected")

    client = Client()
    journal = EventJournal(tmp_path)
    outcome = run_n3_agent(
        client, N3Executor(Transport(), journal, None), max_steps=3,
        system_prompt="system", mission_prompt="mission", initial_context="foothold",
        tool_schemas=tool_schemas(False, 1.25), report_validator=validate_report,
        final_submission_max_tokens=1024, reserved_completion_steps=1,
        episode_deadline_seconds=30,
        budget=EpisodeBudget(EpisodeLimits(max_model_calls=4, max_tool_calls=4,
                                           max_input_tokens=1_000_000, max_output_tokens=10_000)),
        max_context_chars=100_000, context_tail_groups=10,
        decision_recovery_max_output_tokens=256, event_sink=journal.record,
        enable_investigation_truncation_recovery=True,
        force_final_submission_tool_choice=True,
    )
    assert outcome.stop_reason == "submitted_integrated_report"
    assert client.calls[1]["tool_choice"] == "required"
    assert client.calls[1]["max_tokens"] == 256
    assert all(item["function"]["name"] != "submit_integrated_report"
               for item in client.calls[1].get("tools", []))
    assert client.calls[2]["tool_choice"] == {
        "type": "function", "function": {"name": "submit_integrated_report"},
    }
