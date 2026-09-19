"""Three places where a pattern was standing in for a reading, given a reader.

They share one rule: **the judgment only widens what the code already does.** The
pattern runs first and its answer stands; the classifier is asked about what the
pattern let through. A TypeSafe outage, a bad key or a wrong answer therefore leaves
each gate exactly where it was without this module — never looser.

1. **Is this command a merge request, and for which repository** (`MergeRequestJudge`,
   consumed by `mr_gate.migration_gate`). A regex for `glab mr create` does not see
   `git push -o merge_request.create`, an API POST to `merge_requests` or a wrapper
   script, and "the last `cd`" does not see a subshell, a `pushd`, a `-R group/project`.
   The judge answers both; when it cannot tell the repository, every candidate is
   searched. `git diff` still decides whether there is a migration.

2. **Where a working note stands** (`NoteJudge`). A word list (FIXED, RESOLVED, …)
   misses "fix published, waiting for deploy", and the issue is worked again on every
   pass. The judge answers finished / skipped / held / working; only a confident
   non-working answer should stamp a note.

3. **What a test-fixing delegation did to the specs** (`SpecReviewMiddleware`). A
   green suite is not a verified suite: an agent asked to make a red test pass can do
   it by asserting the opposite or by swapping the asserted value. The middleware
   snapshots the suite's specs around the delegation and, for each existing file that
   changed, asks whether an assertion was inverted, retargeted, weakened or disabled;
   sleeps, `console.log`, `.first()` and `test.only` are exact tokens and stay in code.
   The findings are appended to the delegation's result.

What leaves the machine: the shell command (1), the note's text (2), the spec diff (3).
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

RELEASE_STATE = "WAITING-RELEASE"
SKIPPED_STATE = "SKIPPED"
HELD_STATE = "HELD"

__all__ = [
    "MergeRequestJudge",
    "MergeRequestJudgment",
    "NoteJudge",
    "SpecReviewMiddleware",
    "build_mr_judge",
    "build_note_judge",
    "build_spec_review_middleware",
    "candidate_repos",
    "spec_changes",
    "token_findings",
]

# ------------------------------------------------------- 1. the merge request

#: At or above this the command is treated as opening a merge request.
MR_THRESHOLD = 0.5
#: Below this the repository answer is not trusted and every candidate is checked.
REPO_CONFIDENCE = 0.6
UNCLEAR = "unclear"

MAX_COMMAND_CHARS = 8000


@dataclass(frozen=True)
class MergeRequestJudgment:
    """`opens`: probability the command opens a merge request. `repos`: where to look
    for migrations on top of what the patterns found — one repository when the judge
    is sure, every candidate when it is not."""

    opens: float
    repos: tuple[Path, ...] = ()

    @property
    def is_merge_request(self) -> bool:
        return self.opens >= MR_THRESHOLD


def candidate_repos(root: Path) -> dict[str, Path]:
    """The git repositories a command under `root` can be about: `root` and its children."""
    found: dict[str, Path] = {}
    if (root / ".git").exists():
        found[root.name] = root
    try:
        children = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        children = []
    for child in children:
        if (child / ".git").exists():
            found[child.name] = child
    return found


def mr_questions(repos: dict[str, Path]) -> dict[str, Any]:
    from langchain_typesafe import Choice, Noul, NoulCriteria

    questions: dict[str, Any] = {
        "opens_merge_request": Noul(
            instructions=(
                "`command` is a shell command an automated agent is about to run in a "
                "workspace of git repositories hosted on GitLab. Would running it "
                "create a merge request?"
            ),
            criteria=NoulCriteria(
                true="It creates a merge or pull request by any means: `glab mr create`, "
                "a GitLab API call that POSTs to merge_requests, `git push` with the "
                "merge_request.create push option, or a script or alias whose purpose is "
                "opening one.",
                false="It does something else — commits, pushes a branch without push "
                "options, lists, views, updates or merges existing merge requests, "
                "creates an issue — or only mentions merge requests in text it prints "
                "or searches.",
            ),
        )
    }
    if len(repos) > 1:
        criteria = {
            name: f"The command acts on the repository in directory `{name}` ({path}): "
            "it changes into it by any means (cd, pushd, a subshell), passes it to git "
            "-C or to a --repo/-R option, or names its project."
            for name, path in repos.items()
        }
        criteria[UNCLEAR] = (
            "The command does not say which repository it acts on: it relies on the "
            "current directory, a variable or a script."
        )
        questions["repository"] = Choice(
            instructions=(
                "`command` is a shell command about to run in a workspace that holds "
                "several git repositories. Supposing it creates a merge request, for "
                "which repository?"
            ),
            criteria=criteria,
        )
    return questions


class MergeRequestJudge:
    """`judge(command)` → `MergeRequestJudgment`, or `None` when it cannot answer.

    Answers are cached by command: the interrupt middleware calls `when` and then
    `description` with the same call, and a replayed thread calls them again.
    """

    def __init__(self, classifier: Any, repos: dict[str, Path]) -> None:
        self.classifier = classifier
        self.repos = repos
        self._cache: dict[str, MergeRequestJudgment] = {}

    def __call__(self, command: str) -> MergeRequestJudgment | None:
        if not command.strip():
            return None
        if command not in self._cache:
            if len(self._cache) > 256:
                self._cache.clear()
            judgment = self._ask(command)
            if judgment is None:
                # Not cached: an outage must not outlive itself.
                return None
            self._cache[command] = judgment
        return self._cache[command]

    def _ask(self, command: str) -> MergeRequestJudgment | None:
        try:
            response = self.classifier.invoke({"command": command[:MAX_COMMAND_CHARS]})
            opens = response.nouls["opens_merge_request"].noul
        except Exception:  # noqa: BLE001 - no answer: the patterns alone decide, as before
            return None
        if opens < MR_THRESHOLD:
            return MergeRequestJudgment(opens)
        answer = response.choices.get("repository")
        if (
            answer is not None
            and answer.choice in self.repos
            and answer.confidence >= REPO_CONFIDENCE
        ):
            return MergeRequestJudgment(opens, (self.repos[answer.choice],))
        return MergeRequestJudgment(opens, tuple(self.repos.values()))


# ------------------------------------------------------------- 2. the note

#: A stamp stops an issue being worked until a human reverses it, so the judge has
#: to be sure; an unsure answer leaves the note WORKING, which costs one more pass.
NOTE_CONFIDENCE = 0.8
MAX_NOTE_CHARS = 12000

_NOTE_ANSWERS: dict[str, tuple[str | None, str]] = {
    "finished": (
        RELEASE_STATE,
        (
            "The fix is done as far as this team goes: it was committed and published — a "
            "merge request is open or merged, approval was requested or given, or the "
            "issue was resolved in the monitor — and what remains is someone merging or "
            "deploying it."
        ),
    ),
    "skipped": (
        SKIPPED_STATE,
        (
            "The note concludes this is not a defect to fix here: an infrastructure "
            "failure, a validation or business rule working as intended, or a duplicate."
        ),
    ),
    "held": (
        HELD_STATE,
        (
            "The work was stopped by a person or a gate: a reviewer rejected it, a merge "
            "request was refused or is waiting for a human decision the note says not to "
            "work around."
        ),
    ),
    "working": (
        None,
        (
            "The investigation or the fix is still in progress: a root cause being looked "
            "for, a fix not written, not passing or not yet committed and published, or a "
            "next step left for the following pass."
        ),
    ),
}


def note_questions() -> dict[str, Any]:
    from langchain_typesafe import Choice

    return {
        "state": Choice(
            instructions=(
                "`note` is the working note an automated bug-fixing harness keeps about "
                "one production issue, possibly in more than one language, across several "
                "passes; later sections supersede earlier ones. Where does the work on "
                "this issue stand now?"
            ),
            criteria={name: text for name, (_, text) in _NOTE_ANSWERS.items()},
        )
    }


class NoteJudge:
    """`judge(text)` → `(state to stamp or None, confidence)`; never raises."""

    def __init__(self, classifier: Any) -> None:
        self.classifier = classifier

    def __call__(self, text: str) -> tuple[str | None, float]:
        try:
            answer = self.classifier.invoke({"note": text[:MAX_NOTE_CHARS]}).choices["state"]
        except Exception:  # noqa: BLE001 - no answer: the note stays as the words left it
            return None, 0.0
        state = _NOTE_ANSWERS.get(answer.choice, (None, ""))[0]
        if state is None or answer.confidence < NOTE_CONFIDENCE:
            return None, answer.confidence
        return state, answer.confidence


# ------------------------------------------------------------ 3. the specs

SPEC_THRESHOLD = 0.5
MAX_DIFF_CHARS = 12000
MAX_SPEC_FILES = 20
_SPEC_GLOBS = ("*.spec.ts", "*.spec.js", "*.test.ts")

#: What a test-fixing agent must not add. Exact tokens on added lines: code, not judgment.
_TOKENS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("a sleep", re.compile(r"\bwaitForTimeout\s*\(|\bsetTimeout\s*\(")),
    ("console.log", re.compile(r"\bconsole\.log\s*\(")),
    ("`.first()`/`.nth()`/`.last()` on a locator", re.compile(r"\.(?:first|last)\(\)|\.nth\(")),
    ("test.only", re.compile(r"\b(?:test|describe)\.only\s*\(")),
)


@dataclass(frozen=True)
class SpecJudgment:
    name: str
    question: str
    yes: str
    no: str
    finding: str


SPEC_JUDGMENTS: tuple[SpecJudgment, ...] = (
    SpecJudgment(
        "inverted",
        "Does the change make a test assert the opposite of what it asserted before?",
        "An expectation was reversed: visible became hidden, present became absent, "
        "`toBe` became `not.toBe`, success became an error, enabled became disabled, or "
        "the asserted outcome was otherwise swapped for its negation.",
        "Every expectation still asserts the same outcome as before; what changed is "
        "how the page is driven or located, or new expectations were added.",
        "an assertion now expects the opposite",
    ),
    SpecJudgment(
        "weakened",
        "Does the change make an assertion easier to satisfy than before?",
        "An expected value was replaced by a different or less specific one, an exact "
        "matcher by a loose one (toContain, toBeTruthy, a regex), a precise locator by "
        "a broad one, an assertion was deleted, or a timeout or retry count was raised "
        "so the test passes.",
        "The assertions demand at least as much as before; changes fix how the test "
        "reaches the state it checks — a corrected selector for the same element, an "
        "awaited navigation, test data set up properly.",
        "an assertion was weakened or removed",
    ),
    SpecJudgment(
        "retargeted",
        "Does the change make an existing assertion check a different thing than before?",
        "An expectation that existed before now reads a different element, field or "
        "attribute, or expects a different literal value, than it did — a user id "
        "where it checked a tenant id, '43' where it expected '42', another row or "
        "another message.",
        "Every existing expectation still checks the same thing against the same "
        "expected value; only steps that drive the page changed — clicks, fills, "
        "navigation, waits — or expectations were added.",
        "an assertion now checks a different element or value",
    ),
    SpecJudgment(
        "disabled",
        "Does the change stop a test from running or from failing?",
        "A test or block was skipped or marked fixme, returns early, has its "
        "assertions commented out, or wraps them in a try/catch or condition that "
        "swallows the failure.",
        "Every test that ran before still runs and can still fail.",
        "a test was disabled — legitimate only as `test.fixme` with evidence in the report",
    ),
)


def spec_questions() -> dict[str, Any]:
    from langchain_typesafe import Noul, NoulCriteria

    context = (
        "`diff` is a unified diff of one Playwright spec file, changed by an automated "
        "agent asked to make a failing end-to-end test pass by fixing the test's "
        "mechanics or the application — never by changing what the test demands."
    )
    return {
        j.name: Noul(
            instructions=f"{context} {j.question}",
            criteria=NoulCriteria(true=j.yes, false=j.no),
        )
        for j in SPEC_JUDGMENTS
    }


def _snapshot(suite: Path) -> dict[Path, str]:
    files: dict[Path, str] = {}
    if not suite.is_dir():
        return files
    for pattern in _SPEC_GLOBS:
        for path in suite.rglob(pattern):
            if "node_modules" in path.parts:
                continue
            try:
                files[path] = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    return files


def spec_changes(before: dict[Path, str], after: dict[Path, str], suite: Path) -> dict[str, str]:
    """Unified diffs, by suite-relative path, of specs that existed and were changed.

    A spec created during the delegation has no earlier assertion to betray; one that
    was deleted is reported as a diff against nothing, which reads as "disabled".
    """
    changes: dict[str, str] = {}
    for path, old in before.items():
        new = after.get(path, "")
        if new == old:
            continue
        name = str(path.relative_to(suite))
        diff = difflib.unified_diff(
            old.splitlines(), new.splitlines(), f"a/{name}", f"b/{name}", lineterm="", n=3
        )
        changes[name] = "\n".join(diff)
    return changes


def token_findings(diff: str) -> list[str]:
    """Forbidden tokens on the lines a diff adds."""
    added = "\n".join(
        line[1:] for line in diff.splitlines() if line.startswith("+") and line[:3] != "+++"
    )
    return [f"adds {label}" for label, pattern in _TOKENS if pattern.search(added)]


def _render_review(
    changes: dict[str, str], responses: Sequence[Any] | None, error: str | None
) -> str:
    rows: list[str] = []
    for index, (name, diff) in enumerate(changes.items()):
        findings = token_findings(diff)
        response = responses[index] if responses is not None and index < len(responses) else None
        if response is not None and not isinstance(response, BaseException):
            for j in SPEC_JUDGMENTS:
                answer = response.nouls.get(j.name)
                if answer is not None and answer.noul >= SPEC_THRESHOLD:
                    findings.append(f"{j.finding} ({answer.noul:.2f})")
        if findings:
            rows.append(f"- `{name}`: " + "; ".join(findings))
    header = f"## Spec review — {len(changes)} existing spec file(s) changed by this delegation"
    lines = [header, ""]
    if rows:
        lines += [
            (
                "Read these hunks before accepting the table above. A test made green by "
                "demanding less is not fixed: BROKEN-IN-APP keeps its assertion and stays "
                "red; an unrunnable case is a `test.fixme` with evidence. Send back what "
                "does not hold."
            ),
            "",
            *rows,
        ]
    elif error is None:
        lines.append(
            "No assertion judged inverted, retargeted, weakened or disabled, and no forbidden token "
            "added. The status table is still a claim — open the diff of what you accept."
        )
    if error is not None:
        lines += ["", f"[assertion review unavailable: {error} — read every changed spec.]"]
    return "\n".join(lines)


class SpecReviewMiddleware(AgentMiddleware):
    """Append a review of the spec changes to a `fixer` delegation's result."""

    def __init__(self, classifier: Any, suite: Path, *, roles: Sequence[str] = ("fixer",)) -> None:
        """`classifier` is a `TypeSafeClassifier` over `spec_questions()`, or a fake
        with its `batch`/`abatch` and response shape."""
        super().__init__()
        self.classifier = classifier
        self.suite = suite
        self.roles = frozenset(roles)

    def _applies(self, request: ToolCallRequest) -> bool:
        call = request.tool_call
        return (
            call.get("name") == "task"
            and (call.get("args") or {}).get("subagent_type") in self.roles
        )

    def _changes(self, before: dict[Path, str], result: Any) -> dict[str, str]:
        if not isinstance(result, ToolMessage) or not isinstance(result.content, str):
            return {}
        changes = spec_changes(before, _snapshot(self.suite), self.suite)
        return dict(list(changes.items())[:MAX_SPEC_FILES])

    @staticmethod
    def _states(changes: dict[str, str]) -> list[dict[str, str]]:
        return [{"file": name, "diff": diff[:MAX_DIFF_CHARS]} for name, diff in changes.items()]

    @staticmethod
    def _finish(result: ToolMessage, changes: dict[str, str], responses: Any, error: Any):
        failed = [r for r in responses or () if isinstance(r, BaseException)]
        if error is None and failed:
            error = f"{type(failed[0]).__name__}: {failed[0]}"[:200]
        block = _render_review(changes, responses, error)
        return result.model_copy(update={"content": f"{result.content}\n\n{block}"})

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        if not self._applies(request):
            return handler(request)
        before = _snapshot(self.suite)
        result = handler(request)
        changes = self._changes(before, result)
        if not changes:
            return result
        responses, error = None, None
        try:
            responses = self.classifier.batch(self._states(changes), return_exceptions=True)
        except Exception as exc:  # noqa: BLE001 - a review must never fail the delegation
            error = f"{type(exc).__name__}: {exc}"[:200]
        return self._finish(result, changes, responses, error)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if not self._applies(request):
            return await handler(request)
        before = _snapshot(self.suite)
        result = await handler(request)
        changes = self._changes(before, result)
        if not changes:
            return result
        responses, error = None, None
        try:
            responses = await self.classifier.abatch(self._states(changes), return_exceptions=True)
        except Exception as exc:  # noqa: BLE001 - a review must never fail the delegation
            error = f"{type(exc).__name__}: {exc}"[:200]
        return self._finish(result, changes, responses, error)


# ------------------------------------------------------------------ builders


def _classifier(questions: dict[str, Any], **kwargs: Any) -> Any:
    from langchain_typesafe import TypeSafeClassifier

    return TypeSafeClassifier(questions=questions, **kwargs)


def build_mr_judge(root: Path, **classifier_kwargs: Any) -> MergeRequestJudge:
    """A judge over the git repositories at and directly under `root`."""
    repos = candidate_repos(root)
    return MergeRequestJudge(_classifier(mr_questions(repos), **classifier_kwargs), repos)


def build_note_judge(**classifier_kwargs: Any) -> NoteJudge:
    return NoteJudge(_classifier(note_questions(), **classifier_kwargs))


def build_spec_review_middleware(
    suite: Path, *, roles: Sequence[str] = ("fixer",), **classifier_kwargs: Any
) -> SpecReviewMiddleware:
    """`suite` is the directory holding the specs; `roles` the subagents that edit them."""
    return SpecReviewMiddleware(
        _classifier(spec_questions(), **classifier_kwargs), suite, roles=roles
    )
