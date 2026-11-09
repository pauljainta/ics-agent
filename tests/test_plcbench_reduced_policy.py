from ics_agent.plcbench_reduced_policy import parse_reduced_policy_output


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
