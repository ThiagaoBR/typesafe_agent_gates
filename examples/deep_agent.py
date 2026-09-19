"""Everything wired into one Deep Agent. Needs TYPESAFE_API_KEY and a model provider key.

python examples/deep_agent.py "triage today's issues and fix the most urgent defect"
"""

from __future__ import annotations

import sys
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langgraph.checkpoint.memory import InMemorySaver

from typesafe_agent_gates import (
    build_mr_judge,
    build_spec_review_middleware,
    build_toolgate_middleware,
    build_triage_middleware,
    migration_gate,
)

ROOT = Path.cwd()  # a workspace holding one or more git repositories
SUITE = ROOT / "front-end" / "playwright" / "tests"

subagents = [
    {
        "name": "sentry",
        "description": "Lists and briefs production issues from the error monitor.",
        "system_prompt": "You are the error-monitor liaison. Return one briefing per issue id.",
    },
    {
        "name": "fixer",
        "description": "Makes a failing end-to-end spec pass by fixing the app or the spec's mechanics.",
        "system_prompt": "Never change what a test demands. Report a status table.",
        # A subagent's middleware comes only from its own spec: it needs its own gate.
        "middleware": [build_toolgate_middleware("fixer")],
    },
]

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5",
    system_prompt="You coordinate bug fixing. One issue at a time. Report HELD steps as HELD.",
    subagents=subagents,
    backend=LocalShellBackend(root_dir=ROOT),
    middleware=[
        build_triage_middleware(role="sentry"),  # ranks the listing `task(sentry)` returns
        build_spec_review_middleware(SUITE, roles=("fixer",)),  # reviews what `task(fixer)` did
        build_toolgate_middleware("coordinator"),  # judges every `execute`
    ],
    # A merge request that carries a migration waits for a human — however it is opened.
    interrupt_on={"execute": migration_gate(ROOT, base_branch="main", judge=build_mr_judge(ROOT))},
    checkpointer=InMemorySaver(),  # interrupts need one
)

if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "List today's issues."
    config = {"configurable": {"thread_id": "demo"}}
    result = agent.invoke({"messages": [{"role": "user", "content": task}]}, config)
    print(result["messages"][-1].content)
