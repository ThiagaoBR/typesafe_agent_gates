"""Pause a merge request that carries a database migration, for a human.

`migration_gate(...)` returns an `InterruptOnConfig` for LangChain's
`HumanInTheLoopMiddleware` (or deepagents' `interrupt_on`): its `when` predicate fires
only for a shell command that opens a merge request whose branch changes a path
matching `globs` — committed since the target branch, or still on disk.

Detection is patterns first (`glab mr create`, the last `cd`, `--target-branch`). Pass
`judge=judgments.build_mr_judge(root)` and a merge request the patterns miss is caught
too, and the repositories the judge names are searched as well. The judge only widens:
a pattern hit is never overruled, and a judge that fails is a judge that is absent.

`when` runs `git` synchronously; under `langgraph dev` start the server with
`--allow-blocking`.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_MIGRATION_GLOBS",
    "creates_merge_request",
    "migration_changes",
    "migration_gate",
    "repo_of",
    "target_of",
]

DEFAULT_MIGRATION_GLOBS: tuple[str, ...] = ("*migrations/*",)

_MR_CREATE = re.compile(r"\bglab\s+mr\s+create\b")
_TARGET = re.compile(r"--target-branch[= ]+([^\s'\"]+)")
_CD = re.compile(r"(?:^|&&|;|\|\||\n)\s*cd\s+(?:-P\s+)?['\"]?([^\s'\"&;|]+)")
_GIT_C = re.compile(r"\bgit\s+-C\s+['\"]?([^\s'\"&;|]+)")


# ------------------------------------------------------------------ detection


def creates_merge_request(command: str) -> bool:
    """True when a shell command opens a merge request with `glab`."""
    return bool(_MR_CREATE.search(command))


def repo_of(command: str, default: Path) -> Path:
    """The repository a command works in: its last `cd`, else its `git -C`, else `default`."""
    cds = _CD.findall(command)
    if cds:
        return Path(os.path.expanduser(cds[-1]))
    gits = _GIT_C.findall(command)
    if gits:
        return Path(os.path.expanduser(gits[-1]))
    return default


def target_of(command: str, default: str) -> str:
    """The branch the merge request targets, from `--target-branch`, else `default`."""
    match = _TARGET.search(command)
    return match.group(1) if match else default


def _git(repo: Path, *args: str) -> list[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    return [line for line in out.stdout.splitlines() if line.strip()]


def _base_ref(repo: Path, target: str) -> str | None:
    """`origin/<target>` when the remote ref exists, else the local branch, else None."""
    for ref in (f"origin/{target}", target):
        if _git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"):
            return ref
    return None


def migration_changes(repo: Path, target: str, globs: list[str]) -> list[str]:
    """Paths matching `globs` that the branch changes relative to `target`.

    Two sources, both counted: what is committed on the branch since it left the
    target (`<target>...HEAD`, three dots — the branch's own commits, not the
    target's), and what is on disk but not committed — staged, modified or untracked.
    A migration the coordinator wrote and forgot to `git add` is still a migration.
    """
    if not (repo / ".git").exists() and not (repo / ".git").is_file():
        return []
    paths: set[str] = set()
    base = _base_ref(repo, target)
    if base is not None:
        paths.update(_git(repo, "diff", "--name-only", f"{base}...HEAD"))
    for line in _git(repo, "status", "--porcelain", "--untracked-files=all"):
        # `XY path` or `XY old -> new`; the path after the arrow is the live one.
        path = line[3:].split(" -> ")[-1].strip().strip('"')
        if path:
            paths.add(path)
    return sorted(p for p in paths if any(fnmatch.fnmatch(p, g) for g in globs))


def migration_gate(
    root: Path,
    *,
    base_branch: str = "main",
    globs: Sequence[str] = DEFAULT_MIGRATION_GLOBS,
    judge: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """An `InterruptOnConfig` for `execute`: pause a merge request that carries a migration.

    `judge` (`judgments.MergeRequestJudge`, optional) only widens the patterns: a
    command they do not recognise is a merge request when the judge says so, and the
    repositories it names are searched as well as the one `repo_of` found. With no
    judge, or one that cannot answer, this is the patterns alone.
    """

    def _judged(command: str) -> Any:
        if judge is None:
            return None
        try:
            return judge(command)
        except Exception:  # noqa: BLE001 - a judge that fails is a judge that is absent
            return None

    def _is_merge_request(command: str) -> bool:
        if creates_merge_request(command):
            return True
        judged = _judged(command)
        return judged is not None and judged.is_merge_request

    def _files(command: str) -> list[str]:
        repos = [repo_of(command, root)]
        judged = _judged(command)
        if judged is not None and judged.is_merge_request:
            repos += [repo for repo in judged.repos if repo not in repos]
        default = base_branch
        target = target_of(command, default)
        found: set[str] = set()
        for repo in repos:
            targets = [target]
            if target != default and _base_ref(repo, target) is None:
                # `--target-branch "$BASE"`: a target that names no ref would leave
                # only the uncommitted files counted. Measure against the base too.
                targets.append(default)
            for branch in targets:
                found.update(migration_changes(repo, branch, list(globs)))
        return sorted(found)

    def when(request: Any) -> bool:
        command = str((request.tool_call.get("args") or {}).get("command", ""))
        if not _is_merge_request(command):
            return False
        return bool(_files(command))

    def description(tool_call: dict[str, Any], state: Any, runtime: Any) -> str:
        command = str((tool_call.get("args") or {}).get("command", ""))
        files = _files(command)
        listing = "\n".join(f"  - {path}" for path in files) or "  (none found now)"
        return (
            "Merge request with a database migration — a human decides whether it opens.\n\n"
            f"Migration files on the branch:\n{listing}\n\n"
            f"Command:\n  {command.strip()[:1500]}"
        )

    return {
        "allowed_decisions": ["approve", "reject"],
        "when": when,
        "description": description,
    }
