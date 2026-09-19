"""Issue triage: cutting the listing, reading the answers, never failing a delegation.

No network: the classifier is a fake with `TypeSafeClassifier`'s `batch`/`abatch`
and response shape.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from typesafe_agent_gates import triage

LISTING = """\
3 issues in the window, 12 older ones left out.

- SHOP-5G — TypeError: cannot read 'id' of undefined — 4210 events, ongoing
  culprit: reports/stock.js:88, same family as SHOP-7K
- SHOP-7K — Invoice emission fails with HTTP-500 — 6 events, 4 users, regressed
  culprit: billing/invoice/issue.js:212
- SHOP-FRONT-1A2 — ECONNRESET talking to the database — 90 events, escalating
"""


def answer(
    severity: float, urgency: float, kind: str, confidence: float = 0.9, route: str = "back-end"
):
    return SimpleNamespace(
        scores={
            "severity": SimpleNamespace(score=severity),
            "urgency": SimpleNamespace(score=urgency),
        },
        choices={
            "kind": SimpleNamespace(choice=kind, confidence=confidence),
            "route": SimpleNamespace(choice=route, confidence=0.8),
        },
    )


class FakeClassifier:
    def __init__(self, responses=None, *, raises: Exception | None = None):
        self.responses = responses or {}
        self.raises = raises
        self.states: list[dict] = []

    def batch(self, states, *, return_exceptions=False):
        if self.raises is not None:
            raise self.raises
        self.states.extend(states)
        return [self.responses[state["issue"]] for state in states]

    async def abatch(self, states, *, return_exceptions=False):
        return self.batch(states, return_exceptions=return_exceptions)


def request(description="triage since 2026-09-18T00:10", role="sentry", name="task"):
    call = {
        "name": name,
        "id": "call-1",
        "args": {"subagent_type": role, "description": description},
    }
    return SimpleNamespace(tool_call=call)


def result(content=LISTING, status="success"):
    return ToolMessage(content=content, tool_call_id="call-1", name="task", status=status)


RESPONSES = {
    "SHOP-5G": answer(1.1, 0.9, "defect"),
    "SHOP-7K": answer(2.2, 2.8, "defect"),
    "SHOP-FRONT-1A2": answer(2.0, 2.1, "infrastructure", 0.55, route="front-end"),
}


def test_a_listing_is_cut_into_one_briefing_per_issue():
    parts = triage.split_briefings(LISTING)

    assert list(parts) == ["SHOP-5G", "SHOP-7K", "SHOP-FRONT-1A2"]
    # An id mentioned inside another briefing does not open a part of its own there,
    # and words shaped like an id (`HTTP-500`) are not issues.
    assert "same family as SHOP-7K" in parts["SHOP-5G"]
    assert parts["SHOP-7K"].startswith("- SHOP-7K")
    assert "issue.js:212" in parts["SHOP-7K"]
    assert "HTTP-500" not in parts


def test_status_words_are_not_issue_ids():
    text = "SHOP-5G is WAITING-RELEASE, PENDING-APPROVAL; MR NOT-CREATED; NO-ISSUES"

    assert list(triage.split_briefings(text)) == ["SHOP-5G"]


def test_the_table_ranks_by_urgency_then_severity():
    classifier = FakeClassifier(RESPONSES)
    middleware = triage.IssueTriageMiddleware(classifier)

    out = middleware.wrap_tool_call(request(), lambda _: result())

    assert out.content.startswith(LISTING)
    rows = [line for line in out.content.splitlines() if line.startswith("| SHOP")]
    assert [row.split("|")[1].strip() for row in rows] == [
        "SHOP-7K",
        "SHOP-FRONT-1A2",
        "SHOP-5G",
    ]
    assert "now (2.8/3)" in rows[0] and "blocking (2.2/3)" in rows[0]
    assert "infrastructure (0.55)" in rows[1]
    # Where the fix goes is a column of its own: the repository for the branch and MR.
    assert rows[0].endswith("| back-end (0.80) |") and rows[1].endswith("| front-end (0.80) |")
    # The classifier saw each issue's own briefing, by named field.
    assert [state["issue"] for state in classifier.states] == list(RESPONSES)
    assert "issue.js" in classifier.states[1]["briefing"]


def test_the_async_path_ranks_the_same():
    middleware = triage.IssueTriageMiddleware(FakeClassifier(RESPONSES))

    async def handler(_):
        return result()

    out = asyncio.run(middleware.awrap_tool_call(request(), handler))

    assert "| SHOP-7K | now (2.8/3)" in out.content


@pytest.mark.parametrize(
    "req",
    [
        request(role="fixer"),
        request(name="execute"),
        request(description="resolve SHOP-5G in next release"),
        request(description="full briefing of SHOP-7K"),
    ],
)
def test_only_a_triage_delegation_to_the_liaison_is_classified(req):
    classifier = FakeClassifier(RESPONSES)
    original = result()

    out = triage.IssueTriageMiddleware(classifier).wrap_tool_call(req, lambda _: original)

    assert out is original
    assert classifier.states == []


def test_a_failed_delegation_is_left_alone():
    classifier = FakeClassifier(RESPONSES)
    failed = result(status="error")

    out = triage.IssueTriageMiddleware(classifier).wrap_tool_call(request(), lambda _: failed)

    assert out is failed and classifier.states == []


def test_a_classifier_failure_never_fails_the_delegation():
    middleware = triage.IssueTriageMiddleware(FakeClassifier(raises=RuntimeError("401 bad key")))

    out = middleware.wrap_tool_call(request(), lambda _: result())

    assert out.status == "success"
    assert out.content.startswith(LISTING)
    assert "[triage unavailable: RuntimeError: 401 bad key" in out.content
    assert "event count" in out.content


def test_one_failed_request_is_a_row_and_all_failed_is_one_line():
    some = dict(RESPONSES, **{"SHOP-5G": TimeoutError("timed out")})
    out = triage.IssueTriageMiddleware(FakeClassifier(some)).wrap_tool_call(
        request(), lambda _: result()
    )
    rows = [line for line in out.content.splitlines() if line.startswith("| SHOP")]
    assert "not scored" in rows[-1] and "SHOP-5G" in rows[-1]

    every = {issue: TimeoutError("timed out") for issue in RESPONSES}
    out = triage.IssueTriageMiddleware(FakeClassifier(every)).wrap_tool_call(
        request(), lambda _: result()
    )
    assert "| issue |" not in out.content
    assert "[triage unavailable: TimeoutError: timed out" in out.content


def test_a_long_listing_is_scored_up_to_the_cap():
    lines = "\n".join(f"- SHOP-{n:X} — error {n}" for n in range(16, 16 + 40))
    classifier = FakeClassifier({f"SHOP-{n:X}": answer(1, 1, "defect") for n in range(16, 56)})

    out = triage.IssueTriageMiddleware(classifier).wrap_tool_call(
        request(), lambda _: result(lines)
    )

    assert len(classifier.states) == triage.MAX_ISSUES
    assert "10 more issue(s)" in out.content


def test_the_questions_are_the_rubrics():
    pytest.importorskip("langchain_typesafe")
    questions = triage.build_questions()

    assert set(questions) == {"severity", "urgency", "kind", "route"}
    assert set(questions["route"].criteria) == set(triage.ROUTES)
    assert len(questions["severity"].criteria) == len(triage.SEVERITY_LEVELS)
    assert set(questions["kind"].criteria) == set(triage.KINDS)
