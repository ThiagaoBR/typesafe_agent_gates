"""The command gate: what is judged, what closes it, what happens when it is down.

No network: the classifier is a fake with `TypeSafeClassifier`'s `invoke`/`ainvoke`
and response shape. The criteria themselves are measured against the live classifier
by `tools/toolgate_probe.py`, not here.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from typesafe_agent_gates import toolgate


class FakeClassifier:
    def __init__(self, raises: Exception | None = None, **nouls: float):
        self.nouls = nouls
        self.raises = raises
        self.states: list[dict] = []

    def invoke(self, state):
        self.states.append(state)
        if self.raises is not None:
            raise self.raises
        return SimpleNamespace(
            nouls={name: SimpleNamespace(noul=p) for name, p in self.nouls.items()}
        )

    async def ainvoke(self, state):
        return self.invoke(state)


def request(command="npm test", name="execute"):
    args = {"command": command} if name == "execute" else {"file_path": "/backend/x"}
    return SimpleNamespace(tool_call={"name": name, "id": "call-1", "args": args})


def ran(_):
    return ToolMessage(content="ran", tool_call_id="call-1", name="execute")


def gate(classifier, **kwargs):
    return toolgate.ToolGateMiddleware(classifier, role="fixer", **kwargs)


def test_a_command_below_the_threshold_on_every_gate_runs():
    classifier = FakeClassifier(database_write=0.41, production=0.2, destructive=0.1, secrets=0.1)

    out = gate(classifier).wrap_tool_call(request("scripts/test-server.sh deploy x"), ran)

    assert out.content == "ran"
    # Only the command and the role leave the machine — never the conversation.
    assert classifier.states == [{"role": "fixer", "command": "scripts/test-server.sh deploy x"}]


def test_one_gate_at_the_threshold_stops_the_command_and_says_held():
    classifier = FakeClassifier(database_write=0.98, production=0.05, destructive=0.6, secrets=0.0)
    calls = []

    out = gate(classifier).wrap_tool_call(
        request("mysql -e 'DROP TABLE invoices'"), lambda r: calls.append(r) or ran(r)
    )

    assert calls == []
    assert out.status == "error" and out.tool_call_id == "call-1"
    assert "NOT EXECUTED" in out.content
    assert "it would write to a database" in out.content
    assert "it would irreversibly destroy work" in out.content
    assert "database_write 0.98" in out.content
    # The two things the role hears nowhere else: do not route around it, report HELD.
    assert "Do not rephrase" in out.content and "HELD" in out.content


def test_the_threshold_is_a_setting():
    classifier = FakeClassifier(destructive=0.6)

    assert gate(classifier, threshold=0.7).wrap_tool_call(request(), ran).content == "ran"
    assert gate(classifier, threshold=0.5).wrap_tool_call(request(), ran).status == "error"


def test_only_execute_is_judged():
    classifier = FakeClassifier(secrets=1.0)

    out = gate(classifier).wrap_tool_call(request(name="read_file"), ran)

    assert out.content == "ran" and classifier.states == []


def test_an_unreachable_gate_is_closed_unless_told_to_fail_open():
    down = FakeClassifier(raises=TimeoutError("timed out"))

    closed = gate(down).wrap_tool_call(request(), ran)
    assert closed.status == "error"
    assert "could not be reached" in closed.content and "TimeoutError" in closed.content
    assert "HELD" in closed.content

    assert gate(down, fail_closed=False).wrap_tool_call(request(), ran).content == "ran"


def test_the_async_path_gates_the_same():
    async def handler(_):
        return ran(_)

    blocked = asyncio.run(
        gate(FakeClassifier(production=0.9)).awrap_tool_call(request("ssh prod"), handler)
    )
    passed = asyncio.run(gate(FakeClassifier(production=0.1)).awrap_tool_call(request(), handler))
    closed = asyncio.run(
        gate(FakeClassifier(raises=OSError("down"))).awrap_tool_call(request(), handler)
    )

    assert blocked.status == "error" and "production" in blocked.content
    assert passed.content == "ran"
    assert closed.status == "error"


def test_each_gate_is_its_own_question():
    pytest.importorskip("langchain_typesafe")
    questions = toolgate.build_questions()

    assert list(questions) == [g.name for g in toolgate.GATES]
    assert {"database_write", "production", "destructive", "secrets"} == set(questions)
    for question in questions.values():
        assert question.criteria.true and question.criteria.false
