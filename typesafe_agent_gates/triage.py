"""Issue triage by severity and urgency, as typed judgments rather than prose.

An agent that works an error monitor's queue opens with a listing and picks one issue
from it. Left to a prompt, the rule tends to be "the one with the most events" — a
count, because a count is the only thing in a listing a model compares reliably. An
error that fires 4,000 times in a report nobody opens then outranks one that blocks
invoicing for a single customer.

`IssueTriageMiddleware` sits on the coordinator's `task` tool (deepagents' subagent
delegation). When a triage delegation to the monitoring subagent comes back it cuts
the answer into one briefing per issue id, asks TypeSafe's classifier four questions
about each — how severe, how urgent, what kind of thing it is, where the fix goes —
and appends the ranked table to the tool result. The answers are scores on a described
rubric with probabilities, not generated text.

Three things to keep true:

- **It is advisory.** The table orders the queue; the agent's own rules still decide.
- **It never fails the delegation.** The classifier is a network call to a third
  party. Anything it raises becomes one line under the untouched listing.
- **The briefings leave the machine.** They go to the TypeSafe API: keep tokens,
  cookies and personal data out of them.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

__all__ = [
    "KINDS",
    "ROUTES",
    "SEVERITY_LEVELS",
    "URGENCY_LEVELS",
    "IssueTriage",
    "IssueTriageMiddleware",
    "build_questions",
    "build_triage_middleware",
    "render_triage",
    "split_briefings",
]

#: What the questions are told about the application — the classifier sees one
#: briefing and nothing else, and "invoice rejected" means little without it. Pass your own.
DOMAIN = (
    "The application is a multi-tenant business system used daily by paying customers: "
    "invoicing, finance, stock. `briefing` is what the error monitor holds about one "
    "issue in production."
)

#: Ordered rubrics, `(label, what the level looks like)`. The descriptions are the
#: criteria the model reads and must stand on their own; the labels are for the table.
SEVERITY_LEVELS: tuple[tuple[str, str], ...] = (
    (
        "cosmetic",
        (
            "Nothing a user set out to do is prevented: a log-only error, a "
            "visual glitch, a failed background retry that succeeds later."
        ),
    ),
    (
        "degraded",
        (
            "A secondary feature fails or a main one works only with a "
            "workaround; the user's data stays correct."
        ),
    ),
    (
        "blocking",
        (
            "A main flow cannot be completed — a screen does not load, a record "
            "cannot be saved, an invoice cannot be issued — for the users who hit it."
        ),
    ),
    (
        "critical",
        (
            "Data is lost, corrupted or wrongly calculated, one tenant's data is "
            "exposed to another, or the application is down for everyone."
        ),
    ),
)
URGENCY_LEVELS: tuple[tuple[str, str], ...] = (
    (
        "whenever",
        (
            "Old, rare and stable: a handful of events over a long period, not "
            "growing, nobody waiting on it."
        ),
    ),
    (
        "this-week",
        (
            "Ongoing at a steady low rate, or affecting few users who have "
            "another way to finish their work."
        ),
    ),
    (
        "today",
        (
            "New or regressed after a fix, growing, or affecting several users or "
            "tenants in work they do every day."
        ),
    ),
    (
        "now",
        (
            "Escalating quickly or hitting most users right now, or blocking "
            "something with a legal or financial deadline such as issuing an invoice."
        ),
    ),
)
#: What is worth working at all, as a judgment: only a `defect` is.
KINDS: dict[str, str] = {
    "defect": "A bug in the application's own code: a null access, a wrong query, a "
    "missing check, a broken screen. Fixing the code makes it stop.",
    "infrastructure": "The environment failed, not the code: connection timeout, "
    "ECONNRESET, database or external service unavailable, out of memory, a deploy "
    "in progress.",
    "expected": "A validation or business-rule exception doing its job: bad input "
    "refused, a permission denied, an expired session, a rule the user ran into.",
}

#: Where the fix goes — the repository the work branch, the GitLab issue and the merge
#: request belong to. The criteria are the routing table: change where an issue is
#: sent by rewording one, not by teaching a parser another stack-frame layout.
ROUTES: dict[str, str] = {
    "back-end": "The failure is in server code: an API route or controller, a "
    "service, a database query or model, a background job, a fiscal or financial "
    "calculation. Stack frames are Node.js server files; the request is an /api call.",
    "front-end": "The failure is in code running in the browser: a screen, a "
    "component, rendering, form handling, client-side state or navigation. Stack "
    "frames are bundled browser JavaScript or the culprit is a page URL.",
    "both": "The briefing shows the two sides disagreeing about a contract — the "
    "screen sends or expects a field the API does not — so the fix touches both.",
    "unclear": "The briefing does not hold enough to tell where the faulty code is.",
}

#: A Sentry short id: `SHOP-5G`, `SHOP-FRONT-1A2`. The last segment is base 36
#: and short, which is what tells it from `WAITING-RELEASE` or `PENDING-APPROVAL`.
_ISSUE_ID = re.compile(r"\b[A-Z][A-Z0-9]+(?:-[A-Z][A-Z0-9]*)*-[0-9A-Z]{1,5}\b")
#: Upper-case hyphenated words with a short tail that are not issues.
_NOT_AN_ISSUE_PREFIX = (
    "HTTP-",
    "UTF-",
    "ISO-",
    "SHA-",
    "TLS-",
    "NO-",
    "NF-",
    "CT-",
    "X-",
    "BROKEN-",
)

#: What may stand before an id on the line that opens its briefing: `- `, `### 2. `,
#: `| `, `**`, `Issue: `.
_HEADS_LINE = re.compile(r"[\W\d_]*(?:(?:issue|id)\b[\W_]*)?", re.IGNORECASE)

#: How a delegation says it is the listing and not a resolve, a status or the full
#: briefing of the one issue already chosen ("triage", "triagem").
_TRIAGE_WORD = re.compile(r"triag", re.IGNORECASE)

#: One request per issue; a day's window rarely holds more, and a listing that does
#: is ranked on its head rather than paid for whole.
MAX_ISSUES = 30
#: Characters of one briefing sent to the classifier.
MAX_BRIEFING_CHARS = 6000


@dataclass(frozen=True)
class IssueTriage:
    """The three answers about one issue; `error` instead when its request failed."""

    issue: str
    severity: float | None = None
    urgency: float | None = None
    kind: str | None = None
    kind_confidence: float | None = None
    route: str | None = None
    route_confidence: float | None = None
    error: str | None = None


def _first_id(line: str) -> re.Match[str] | None:
    return next(
        (m for m in _ISSUE_ID.finditer(line) if not m[0].startswith(_NOT_AN_ISSUE_PREFIX)),
        None,
    )


def split_briefings(text: str) -> dict[str, str]:
    """Cut a liaison's answer into `{issue id: its part of the text}`, in order.

    A part starts at the first line an id *heads* — nothing before it but list or
    heading markup, a number, or an `id:` label — and runs to the next such line, so
    an id mentioned inside another issue's briefing ("same family as SHOP-7K") does
    not cut it short. A listing written some other way, where no id heads a line,
    falls back to the first id on each line.
    """
    lines = text.splitlines()
    found = [(number, match) for number, line in enumerate(lines) if (match := _first_id(line))]
    heading = [(n, m) for n, m in found if _HEADS_LINE.fullmatch(m.string[: m.start()])]
    starts: dict[str, int] = {}
    for number, match in heading or found:
        starts.setdefault(match[0], number)
    ordered = sorted(starts.items(), key=lambda item: item[1])
    ends = [start for _, start in ordered[1:]] + [len(lines)]
    return {
        issue: "\n".join(lines[start:end]).strip()
        for (issue, start), end in zip(ordered, ends, strict=True)
    }


def build_questions(domain: str = DOMAIN) -> dict[str, Any]:
    """The classifier's questions. Imports the optional package, so call it late."""
    from langchain_typesafe import Choice, Score

    return {
        "severity": Score(
            instructions=f"{domain} How bad is the impact on a user who hits this issue?",
            criteria=[description for _, description in SEVERITY_LEVELS],
        ),
        "urgency": Score(
            instructions=(
                f"{domain} How soon does this issue need someone working on it, going "
                "by its event count, how many users it reaches, its first and last "
                "seen dates and its substatus (new, ongoing, escalating, regressed)?"
            ),
            criteria=[description for _, description in URGENCY_LEVELS],
        ),
        "kind": Choice(
            instructions=f"{domain} What kind of problem is this issue?",
            criteria=dict(KINDS),
        ),
        "route": Choice(
            instructions=(
                f"{domain} Supposing the issue is a defect in the application's code, "
                "in which part of the code base is the fix?"
            ),
            criteria=dict(ROUTES),
        ),
    }


def _state(issue: str, briefing: str) -> dict[str, str]:
    return {"issue": issue, "briefing": briefing[:MAX_BRIEFING_CHARS]}


def _read(issue: str, response: Any) -> IssueTriage:
    if isinstance(response, BaseException):
        return IssueTriage(issue, error=f"{type(response).__name__}: {response}"[:200])
    kind = response.choices.get("kind")
    route = response.choices.get("route")
    severity = response.scores.get("severity")
    urgency = response.scores.get("urgency")
    return IssueTriage(
        issue,
        severity=severity.score if severity else None,
        urgency=urgency.score if urgency else None,
        kind=kind.choice if kind else None,
        kind_confidence=kind.confidence if kind else None,
        route=route.choice if route else None,
        route_confidence=route.confidence if route else None,
    )


def _level(score: float | None, levels: Sequence[tuple[str, str]]) -> str:
    if score is None:
        return "?"
    label = levels[min(max(round(score), 0), len(levels) - 1)][0]
    return f"{label} ({score:.1f}/{len(levels) - 1})"


def _choice(choice: str | None, confidence: float | None) -> str:
    return f"{choice or '?'}" + (f" ({confidence:.2f})" if confidence is not None else "")


def render_triage(results: Sequence[IssueTriage], *, left_out: int = 0) -> str:
    """The block appended to the listing: most urgent first, then most severe."""
    ranked = sorted(
        results,
        key=lambda r: (r.error is not None, -(r.urgency or 0.0), -(r.severity or 0.0)),
    )
    rows = [
        "## Triage by urgency and severity",
        "",
        (
            "Typed judgments on each briefing above, most urgent first. Work the first row "
            "that is a `defect` and is not parked; a row judged `infrastructure` or "
            "`expected` with low confidence is worth a look before it is skipped. This "
            "orders the queue — the parked list and the notes still decide what is worked. "
            "`fix in` is where the faulty code most likely is — the repository that takes "
            "the work branch, the GitLab issue and the merge request; the stack frames "
            "in the full briefing settle it."
        ),
        "",
        "| issue | urgency | severity | kind (confidence) | fix in (confidence) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in ranked:
        if r.error is not None:
            rows.append(f"| {r.issue} | not scored | not scored | {r.error} | |")
            continue
        rows.append(
            f"| {r.issue} | {_level(r.urgency, URGENCY_LEVELS)} | "
            f"{_level(r.severity, SEVERITY_LEVELS)} | {_choice(r.kind, r.kind_confidence)} | "
            f"{_choice(r.route, r.route_confidence)} |"
        )
    if left_out:
        rows.append("")
        rows.append(f"{left_out} more issue(s) in the listing were not scored (cap {MAX_ISSUES}).")
    return "\n".join(rows)


def _unavailable(exc: BaseException) -> str:
    detail = str(exc).strip().splitlines()[0][:300] if str(exc).strip() else ""
    return (
        f"[triage unavailable: {type(exc).__name__}: {detail} — order the issues by "
        "event count, as without it.]"
    )


class IssueTriageMiddleware(AgentMiddleware):
    """Append a severity/urgency ranking to the Sentry liaison's triage listing."""

    def __init__(self, classifier: Any, *, role: str = "sentry") -> None:
        """`classifier` is a `TypeSafeClassifier` over `build_questions()`, or any
        runnable with its `batch`/`abatch` and response shape — tests pass a fake."""
        super().__init__()
        self.classifier = classifier
        self.role = role

    def _briefings(self, request: ToolCallRequest, result: Any) -> dict[str, str]:
        call = request.tool_call
        args = call.get("args") or {}
        if call.get("name") != "task" or args.get("subagent_type") != self.role:
            return {}
        if not _TRIAGE_WORD.search(str(args.get("description", ""))):
            return {}
        if not isinstance(result, ToolMessage) or result.status == "error":
            return {}
        return split_briefings(result.content) if isinstance(result.content, str) else {}

    @staticmethod
    def _append(result: ToolMessage, block: str) -> ToolMessage:
        return result.model_copy(update={"content": f"{result.content}\n\n{block}"})

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        result = handler(request)
        briefings = self._briefings(request, result)
        if not briefings:
            return result
        issues = list(briefings)[:MAX_ISSUES]
        try:
            responses = self.classifier.batch(
                [_state(issue, briefings[issue]) for issue in issues], return_exceptions=True
            )
        except Exception as exc:  # noqa: BLE001 - triage must never fail the delegation
            return self._append(result, _unavailable(exc))
        return self._append(result, self._render(issues, responses, len(briefings)))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        result = await handler(request)
        briefings = self._briefings(request, result)
        if not briefings:
            return result
        issues = list(briefings)[:MAX_ISSUES]
        try:
            responses = await self.classifier.abatch(
                [_state(issue, briefings[issue]) for issue in issues], return_exceptions=True
            )
        except Exception as exc:  # noqa: BLE001 - triage must never fail the delegation
            return self._append(result, _unavailable(exc))
        return self._append(result, self._render(issues, responses, len(briefings)))

    @staticmethod
    def _render(issues: list[str], responses: Sequence[Any], total: int) -> str:
        results = [
            _read(issue, response) for issue, response in zip(issues, responses, strict=True)
        ]
        failures = [r for r in results if r.error is not None]
        if failures and len(failures) == len(results):
            # Every request failed the same way (a bad key, the service down): one
            # line says it better than a table of errors.
            return f"[triage unavailable: {failures[0].error} — order the issues by event count, as without it.]"
        return render_triage(results, left_out=total - len(issues))


def build_triage_middleware(
    *, role: str = "sentry", domain: str = DOMAIN, **classifier_kwargs: Any
) -> IssueTriageMiddleware:
    """The middleware over a live `TypeSafeClassifier` (reads `TYPESAFE_API_KEY`)."""
    from langchain_typesafe import TypeSafeClassifier

    classifier = TypeSafeClassifier(questions=build_questions(domain), **classifier_kwargs)
    return IssueTriageMiddleware(classifier, role=role)
