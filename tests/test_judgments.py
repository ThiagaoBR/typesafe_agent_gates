"""The three judgments that widen a pattern, and the rule that they only widen.

No network: every classifier is a fake with `TypeSafeClassifier`'s response shape. The
questions themselves are measured live by `tools/judgments_probe.py`.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from typesafe_agent_gates import judgments, mr_gate


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def _repo(path: Path, *, migration: bool) -> Path:
    path.mkdir()
    _git(path, "init", "-q", "-b", "daily")
    _git(path, "config", "user.email", "qa@example.test")
    _git(path, "config", "user.name", "qa")
    (path / "app.ts").write_text("export const x = 1;\n")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "base")
    _git(path, "switch", "-q", "-c", "work")
    if migration:
        (path / "migrations").mkdir()
        (path / "migrations" / "001_add.sql").write_text("ALTER TABLE x ADD y INT;\n")
        _git(path, "add", ".")
        _git(path, "commit", "-q", "-m", "migration")
    return path


@pytest.fixture
def tree(tmp_path: Path):
    """Two repositories under the scope: the back end carries a migration."""
    back = _repo(tmp_path / "back-end", migration=True)
    front = _repo(tmp_path / "front-end", migration=False)
    return tmp_path, back, front


def _request(command: str) -> SimpleNamespace:
    return SimpleNamespace(tool_call={"name": "execute", "args": {"command": command}})


class MrClassifier:
    def __init__(self, opens=0.0, repo=None, confidence=0.9, raises=None):
        self.opens, self.repo, self.confidence, self.raises = opens, repo, confidence, raises
        self.calls = 0

    def invoke(self, state):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        choices = {}
        if self.repo is not None:
            choices["repository"] = SimpleNamespace(choice=self.repo, confidence=self.confidence)
        return SimpleNamespace(
            nouls={"opens_merge_request": SimpleNamespace(noul=self.opens)}, choices=choices
        )


# ----------------------------------------------------------- 1. merge request


def test_a_merge_request_the_pattern_misses_is_gated_when_the_judge_sees_it(tree):
    root, back, _front = tree
    command = f"(pushd {back} >/dev/null; git push -o merge_request.create origin work)"

    # The patterns alone: not `glab mr create`, so no gate — the hole being closed.
    assert mr_gate.migration_gate(root, base_branch="daily")["when"](_request(command)) is False

    judge = judgments.MergeRequestJudge(
        MrClassifier(opens=0.97, repo="back-end"), judgments.candidate_repos(root)
    )
    gate = mr_gate.migration_gate(root, base_branch="daily", judge=judge)
    assert gate["when"](_request(command)) is True
    assert "migrations/001_add.sql" in gate["description"](
        {"args": {"command": command}}, None, None
    )
    # `when` and `description` share one answer.
    assert judge.classifier.calls == 1


def test_an_unsure_repository_means_every_repository_is_searched(tree):
    root, _back, _front = tree
    repos = judgments.candidate_repos(root)
    command = "./scripts/open-mr.sh"

    unsure = judgments.MergeRequestJudge(MrClassifier(0.9, "front-end", confidence=0.4), repos)
    unclear = judgments.MergeRequestJudge(MrClassifier(0.9, judgments.UNCLEAR), repos)
    sure_front = judgments.MergeRequestJudge(MrClassifier(0.9, "front-end"), repos)

    assert (
        mr_gate.migration_gate(root, base_branch="daily", judge=unsure)["when"](_request(command))
        is True
    )
    assert (
        mr_gate.migration_gate(root, base_branch="daily", judge=unclear)["when"](_request(command))
        is True
    )
    # Confidently the repository without a migration: nothing to gate.
    assert (
        mr_gate.migration_gate(root, base_branch="daily", judge=sure_front)["when"](
            _request(command)
        )
        is False
    )


def test_the_judge_only_widens_the_patterns(tree):
    root, back, _front = tree
    mr = f"cd {back} && glab mr create --target-branch daily --yes"

    # The judge says "not a merge request": the pattern's answer stands.
    denies = judgments.MergeRequestJudge(MrClassifier(opens=0.01), judgments.candidate_repos(root))
    assert (
        mr_gate.migration_gate(root, base_branch="daily", judge=denies)["when"](_request(mr))
        is True
    )

    # The judge is down: exactly the behaviour without it, and the failure is not cached.
    down = judgments.MergeRequestJudge(MrClassifier(raises=TimeoutError("x")), {})
    gate = mr_gate.migration_gate(root, base_branch="daily", judge=down)
    assert gate["when"](_request(mr)) is True
    assert gate["when"](_request("git push origin work")) is False
    assert down("git push origin work") is None and down.classifier.calls >= 2

    # An ordinary command the judge clears is not gated.
    assert (
        mr_gate.migration_gate(root, base_branch="daily", judge=denies)["when"](
            _request("git status")
        )
        is False
    )


def test_a_target_that_names_no_ref_is_also_measured_against_the_base(tree):
    root, back, _front = tree
    command = f'cd {back} && glab mr create --target-branch "$BASE" --yes'

    assert mr_gate.migration_gate(root, base_branch="daily")["when"](_request(command)) is True


def test_candidate_repositories_are_the_scope_and_its_children(tree):
    root, back, front = tree

    assert judgments.candidate_repos(root) == {"back-end": back, "front-end": front}


# ------------------------------------------------------------------ 2. notes


class NoteClassifier:
    def __init__(self, choice="working", confidence=0.95, raises=None):
        self.choice, self.confidence, self.raises = choice, confidence, raises

    def invoke(self, state):
        if self.raises is not None:
            raise self.raises
        return SimpleNamespace(
            choices={"state": SimpleNamespace(choice=self.choice, confidence=self.confidence)}
        )


NOTE = "# SHOP-77\n\nFix published in MR !131, waiting for the production deploy.\n"


def test_a_confident_answer_is_the_state_to_stamp():
    assert judgments.NoteJudge(NoteClassifier("finished", 0.93))(NOTE) == ("WAITING-RELEASE", 0.93)
    assert judgments.NoteJudge(NoteClassifier("skipped", 0.9))(NOTE)[0] == "SKIPPED"
    assert judgments.NoteJudge(NoteClassifier("held", 0.9))(NOTE)[0] == "HELD"


@pytest.mark.parametrize(
    "classifier",
    [
        NoteClassifier("working", 0.99),
        NoteClassifier("finished", 0.6),  # not sure enough to stop the work
        NoteClassifier(raises=TimeoutError("down")),
    ],
)
def test_an_unsure_or_absent_judge_stamps_nothing(classifier):
    assert judgments.NoteJudge(classifier)(NOTE)[0] is None


# ------------------------------------------------------------------ 3. specs

SPEC = """\
import { test, expect } from '@playwright/test';

test('logout is offered', async ({ page }) => {
  await page.goto('/painel');
  await expect(page.getByRole('button', { name: 'Sair' })).toBeVisible();
});
"""
INVERTED = SPEC.replace(").toBeVisible();", ").toBeHidden();\n  await page.waitForTimeout(2000);")


class SpecClassifier:
    def __init__(self, raises=None, **nouls):
        self.nouls, self.raises = nouls, raises
        self.states: list[dict] = []

    def batch(self, states, *, return_exceptions=False):
        if self.raises is not None:
            raise self.raises
        self.states.extend(states)
        return [
            SimpleNamespace(nouls={k: SimpleNamespace(noul=v) for k, v in self.nouls.items()})
            for _ in states
        ]

    async def abatch(self, states, *, return_exceptions=False):
        return self.batch(states, return_exceptions=return_exceptions)


def _task(role="fixer"):
    call = {"name": "task", "id": "c1", "args": {"subagent_type": role, "description": "fix"}}
    return SimpleNamespace(tool_call=call)


def _fixer(suite: Path, *writes: tuple[str, str]):
    def handler(_):
        for name, content in writes:
            (suite / name).write_text(content)
        return ToolMessage(content="| logout | FIXED |", tool_call_id="c1", name="task")

    return handler


@pytest.fixture
def suite(tmp_path: Path) -> Path:
    tests = tmp_path / "playwright" / "tests"
    tests.mkdir(parents=True)
    (tests / "auth.spec.ts").write_text(SPEC)
    return tests


def test_a_spec_made_green_by_asserting_the_opposite_is_reported(suite):
    classifier = SpecClassifier(inverted=0.96, weakened=0.2, disabled=0.02)
    middleware = judgments.SpecReviewMiddleware(classifier, suite)

    out = middleware.wrap_tool_call(_task(), _fixer(suite, ("auth.spec.ts", INVERTED)))

    assert out.content.startswith("| logout | FIXED |")
    assert "## Spec review — 1 existing spec file(s) changed" in out.content
    assert "`auth.spec.ts`" in out.content
    assert "an assertion now expects the opposite (0.96)" in out.content
    assert "weakened" not in out.content
    # The sleep is an exact token: found by code, with or without the classifier.
    assert "adds a sleep" in out.content
    assert "Read these hunks before accepting" in out.content
    # The classifier saw the diff of that file, not the file or the conversation.
    (state,) = classifier.states
    assert state["file"] == "auth.spec.ts"
    assert "-  await expect" in state["diff"] and "+  await expect" in state["diff"]


def test_a_new_spec_and_an_untouched_suite_are_not_reviewed(suite):
    classifier = SpecClassifier(inverted=1.0)
    middleware = judgments.SpecReviewMiddleware(classifier, suite)

    out = middleware.wrap_tool_call(_task(), _fixer(suite, ("new.spec.ts", SPEC)))

    assert out.content == "| logout | FIXED |"
    assert classifier.states == []


def test_only_a_fixer_delegation_is_reviewed(suite):
    classifier = SpecClassifier(inverted=1.0)
    middleware = judgments.SpecReviewMiddleware(classifier, suite)

    out = middleware.wrap_tool_call(_task("generator"), _fixer(suite, ("auth.spec.ts", INVERTED)))

    assert "Spec review" not in out.content and classifier.states == []


def test_a_clean_change_says_so_and_a_dead_classifier_never_fails_the_delegation(suite):
    fixed = SPEC.replace("'/painel'", "'/painel/inicio'")
    clean = judgments.SpecReviewMiddleware(
        SpecClassifier(inverted=0.02, weakened=0.1, disabled=0.0), suite
    ).wrap_tool_call(_task(), _fixer(suite, ("auth.spec.ts", fixed)))
    assert "No assertion judged inverted, retargeted, weakened or disabled" in clean.content

    (suite / "auth.spec.ts").write_text(SPEC)
    down = judgments.SpecReviewMiddleware(SpecClassifier(raises=OSError("down")), suite)
    out = down.wrap_tool_call(_task(), _fixer(suite, ("auth.spec.ts", INVERTED)))
    assert out.status == "success"
    assert "adds a sleep" in out.content
    assert "[assertion review unavailable: OSError: down" in out.content


def test_the_async_path_reviews_the_same(suite):
    middleware = judgments.SpecReviewMiddleware(SpecClassifier(weakened=0.8), suite)

    async def handler(request):
        return _fixer(suite, ("auth.spec.ts", INVERTED))(request)

    out = asyncio.run(middleware.awrap_tool_call(_task(), handler))

    assert "an assertion was weakened or removed (0.80)" in out.content


def test_tokens_are_read_on_added_lines_only():
    diff = "--- a/x\n+++ b/x\n-  console.log('old');\n+  await row.first().click();\n context\n"

    assert judgments.token_findings(diff) == ["adds `.first()`/`.nth()`/`.last()` on a locator"]


# --------------------------------------------------------------- the questions


def test_the_questions_build():
    pytest.importorskip("langchain_typesafe")
    repos = {"back-end": Path("/w/back-end"), "front-end": Path("/w/front-end")}
    questions = judgments.mr_questions(repos)
    assert set(questions["repository"].criteria) == {*repos, judgments.UNCLEAR}
    assert "repository" not in judgments.mr_questions({"only": Path("/w")})
    assert set(judgments.note_questions()["state"].criteria) == {
        "finished",
        "skipped",
        "held",
        "working",
    }
    assert set(judgments.spec_questions()) == {"inverted", "retargeted", "weakened", "disabled"}
