from ics_agent.plcbench_reduced_policy import _sandbox_command_kind, parse_reduced_policy_output


def test_reduced_policy_accepts_action_without_reasoning_fields():
    assert parse_reduced_policy_output("Action: exec_python\n'''\nprint('x')\n'''") == {
        "kind": "exec_python", "payload": "print('x')"
    }


def test_reduced_policy_accepts_submit_without_reasoning_fields():
    assert parse_reduced_policy_output("Action: submit\n'''\ndone\n'''") == {
        "kind": "final", "summary": "done"
    }


def test_reduced_policy_rejects_unknown_interface():
    assert parse_reduced_policy_output("Action: arbitrary_tool\n'''\nx\n'''") == {
        "kind": "noop", "reason": "unknown_tool"
    }


def test_reduced_policy_preserves_normalized_sandbox_command_kinds():
    # NativeEpisodeRunner normalizes exec_python/exec_shell before dispatch.
    assert _sandbox_command_kind("exec_python") == "python"
    assert _sandbox_command_kind("python") == "python"
    assert _sandbox_command_kind("exec_shell") == "shell"
    assert _sandbox_command_kind("shell") == "shell"
