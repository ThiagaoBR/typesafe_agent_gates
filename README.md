# typesafe-agent-gates

Typed judgments from [TypeSafe](https://docs.typesafe.ai/)'s System One model (Jev) as
[LangChain](https://docs.langchain.com/oss/python/langchain/middleware/overview) /
[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) middleware, for
an unattended coding agent: the places where a regex, a word list or a line in a
prompt was standing in for *reading* something.

Extracted from a harness that works a Sentry queue overnight — triage, regression
spec, fix, merge request — with nobody watching. Every question here was measured
against the live classifier with a labelled probe (`tools/`), not assumed.

| piece | what it replaces | hook |
| --- | --- | --- |
| [`ToolGateMiddleware`](typesafe_agent_gates/toolgate.py) | a blocklist for shell commands | `wrap_tool_call` on `execute` |
| [`IssueTriageMiddleware`](typesafe_agent_gates/triage.py) | "pick the issue with the most events" | `wrap_tool_call` on `task` → monitoring subagent |
| [`MergeRequestJudge`](typesafe_agent_gates/judgments.py) + [`migration_gate`](typesafe_agent_gates/mr_gate.py) | `\bglab\s+mr\s+create\b` and "the last `cd`" | `interrupt_on["execute"]["when"]` |
| [`NoteJudge`](typesafe_agent_gates/judgments.py) | a status word list (FIXED, RESOLVED, …) | plain callable |
| [`SpecReviewMiddleware`](typesafe_agent_gates/judgments.py) | "read the spec before accepting the table" in a prompt | `wrap_tool_call` on `task` → test-fixing subagent |

## Install

```bash
git clone <this repo> && cd typesafe-agent-gates
uv venv && uv pip install -e ".[dev]"        # or: pip install -e ".[dev]"
cp .env.example .env                         # TYPESAFE_API_KEY=…
```

- Python ≥ 3.11, `langchain>=1.4`, `langgraph>=1.2`,
  [`langchain-typesafe`](https://pypi.org/project/langchain-typesafe/) (an **alpha**, `0.0.1a2`
  when this was written — pin what you test against).
- A key from the [TypeSafe console](https://console.typesafe.ai/settings/keys) in
  `TYPESAFE_API_KEY`. `TYPESAFE_BASE_URL` points at a gateway or private deployment.
- `examples/deep_agent.py` additionally needs `deepagents` and a model provider key.

```bash
pytest -q                                    # 36 tests, no network (fake classifiers)
python tools/toolgate_probe.py               # 27 labelled commands, live
python tools/judgments_probe.py              # 31 labelled cases, live
```

## The command gate

A shell tool cannot be gated by pattern: `mysql -e "DROP …"`,
`npx sequelize db:migrate:undo:all` and `node -e "…query('DELETE …')"` share no token,
and a blocklist that tried would also stop `rg "DROP TABLE" migrations/`.

```python
from typesafe_agent_gates import build_toolgate_middleware

agent = create_deep_agent(..., middleware=[build_toolgate_middleware("coordinator")])
# a subagent's middleware comes only from its own spec — give each one with a shell its own:
subagent = {"name": "fixer", ..., "middleware": [build_toolgate_middleware("fixer")]}
```

Four independent yes/no questions ([Noul](https://docs.typesafe.ai/primitives/noul)) in
one request, ~0.3 s:

| gate | closes on |
| --- | --- |
| `database_write` | a client, script or ORM CLI that inserts, updates, deletes, drops, alters, runs or undoes migrations, seeds, restores a dump |
| `production` | ssh/scp, a production host or database, kubectl, a remote docker context, a deploy script |
| `destructive` | `rm -rf` on source, `push --force`, `reset --hard`, `clean -fd`, deleted branches, `curl … \| bash` |
| `secrets` | reading `.env`, keys or tokens, printing the environment, sending any of it out |

One answer ≥ `threshold` (0.5) and the command does not run; the agent gets an error
`ToolMessage` naming the judgment, told not to rephrase it through and to report the
step as **HELD**. Design choices worth keeping:

- **Separate questions, not one risk score** — "any serious violation" needs a
  condition per violation ([composite scoring](https://docs.typesafe.ai/patterns/composite-scoring)).
- **Only `{role, command}` is sent**, never the conversation. `langchain-typesafe`'s own
  [`AutoModeMiddleware`](https://docs.langchain.com/oss/python/integrations/providers/typesafe#tool-risk-gating)
  sends the last 30 messages and asks whether the *user* authorised the call; in an
  unattended run nobody is a user.
- **Closed when TypeSafe is unreachable** (`fail_closed=True`): nothing runs, the run
  ends with HELD steps instead of hanging. `fail_closed=False` trades that for availability.
- **`CONTEXT` describes the agent's legitimate work.** Without it `git push` is an
  "external side effect" and the agent stops at its first commit. Pass your own
  `context=`.

Measured: 27/27 on `jev-1.13.0` — reads, tests, commits, pushes, MR creation, a local
test-server deploy and `rg 'DROP TABLE'` pass; every write, production, destructive and
secrets command is blocked. This is a second layer, not a boundary: credentials the
agent must never use still do not belong in its environment.

## Issue triage and routing

```python
from typesafe_agent_gates import build_triage_middleware

middleware = [build_triage_middleware(role="sentry", domain="<what your application is>")]
```

When `task(subagent_type="sentry", description="…triage…")` returns, the answer is cut
into one briefing per issue id and each gets four questions in one request:
**severity** and **urgency** ([Score](https://docs.typesafe.ai/primitives/score) rubrics),
**kind** — defect / infrastructure / expected — and **fix in** — back-end / front-end /
both / unclear ([Choice](https://docs.typesafe.ai/primitives/choice)). The ranked table
is appended to the tool result:

```
| issue | urgency | severity | kind (confidence) | fix in (confidence) |
| SHOP-7K | now (3.0/3) | blocking (2.0/3) | defect (0.97) | back-end (1.00) |
| SHOP-FRONT-1A2 | now (2.8/3) | blocking (2.0/3) | infrastructure (1.00) | back-end (0.99) |
| SHOP-5G | this-week (1.4/3) | cosmetic (0.0/3) | defect (0.99) | back-end (0.98) |
```

A 46-event invoicing blocker outranks a 4,210-event background error that retries
fine — the opposite of "most events". Routing is the `ROUTES` criteria: change where an
issue goes by rewording one, not by teaching a parser another stack-frame layout.
Advisory, and a classifier failure is one line under the untouched listing. 0.87 s for
four issues, ~1,000 input tokens each.

## Judgments that widen a pattern

One rule for all three: **the pattern runs first and its answer stands; the judgment is
asked about what the pattern let through.** An outage or a wrong answer leaves each
gate where it was — never looser.

**A merge request, however it is opened.** `migration_gate` pauses (human in the loop)
a merge request whose branch carries a database migration. With a judge it also sees
`git push -o merge_request.create`, a `…/merge_requests` API POST, `curl`, a wrapper
script — and which repository (`pushd`, a subshell, `-R group/project`); unsure means
every repository under the root is searched. `git diff` still decides whether there is
a migration.

```python
from typesafe_agent_gates import build_mr_judge, migration_gate

interrupt_on = {"execute": migration_gate(ROOT, base_branch="main", judge=build_mr_judge(ROOT))}
```

**Where a working note stands.** `build_note_judge()(text)` → `(state | None, confidence)`
with state one of `WAITING-RELEASE`, `SKIPPED`, `HELD`; `None` means still working or
not sure enough (< 0.8). For notes an agent keeps between runs, where "fix published,
waiting for deploy" matches no status word and the issue gets worked again every pass.

**What a test-fixing delegation did to the specs.** A green suite is not a verified
suite: asked to make a red test pass, an agent asserted the opposite (a logout button
*absent*) and swapped the asserted value (user id for tenant id), then reported PASS.
`SpecReviewMiddleware` snapshots the suite around `task(fixer)` and judges every
*existing* spec that changed — assertion **inverted**, **retargeted**, **weakened**,
test **disabled** — while sleeps, `console.log`, `.first()` and `test.only` stay exact
tokens in code. Findings are appended under the subagent's status table.

```python
from typesafe_agent_gates import build_spec_review_middleware

middleware = [build_spec_review_middleware(Path("front-end/playwright/tests"), roles=("fixer",))]
```

Measured: 31/31. The first run was 29/30 — "tenant id swapped for user id" scored 0.29
as *weakened*, which is why *retargeted* is a question of its own. That is the argument
for the probes: a criterion is prompt text, and only a labelled run says what it does.

## What leaves your machine

Shell commands (gate, MR judge), issue briefings (triage), note text, spec diffs — sent
to the TypeSafe API. Never the conversation. Keep credentials and personal data out of
all of them; TypeSafe's own warning applies: do not put secrets in state unless sending
them is acceptable.

## Tracing

`TypeSafeClassifier` is a LangChain `Runnable`: with `LANGSMITH_TRACING=true` every
classification — state, answers, token usage — shows up in
[LangSmith](https://docs.langchain.com/langsmith/home) next to the agent run.

## The skill

`skills/typesafe-ai/` is TypeSafe's own agent skill (MIT, from
[typesafe-ai/skills](https://github.com/typesafe-ai/skills)), vendored so a coding agent
working on this repository designs new judgments the way these were designed. Prefer
the upstream copy:

```bash
# Claude Code
claude plugin marketplace add typesafe-ai/skills
claude plugin install typesafe@typesafe-ai      # then: /typesafe:typesafe-ai
# or by hand, from this repository
cp -r skills/typesafe-ai ~/.claude/skills/
```

Other agents and updates: TypeSafe's [installation guide](https://docs.typesafe.ai/agent-skill#installation).

What it asks for, and what this repository did with it: read the live docs; keep
rules, lookups and execution in code and ask the model only for the judgment; one
narrow question per judgment, asked together over the same state; a Noul near 0.5 is
an even split, not "medium" — use a Score for a spectrum; include a no-match outcome
(`unclear`, `working`); evaluate thresholds on your own data.

## Layout

```
typesafe_agent_gates/
  toolgate.py     ToolGateMiddleware — four gates on a shell tool
  triage.py       IssueTriageMiddleware — severity, urgency, kind, route
  judgments.py    MergeRequestJudge, NoteJudge, SpecReviewMiddleware
  mr_gate.py      migration_gate — the interrupt config the MR judge plugs into
tools/            live probes: labelled cases against the real classifier
tests/            wiring tests with fake classifiers (no network)
examples/         everything wired into one Deep Agent
skills/           TypeSafe's agent skill (MIT, vendored)
```

## References

TypeSafe
- [Documentation](https://docs.typesafe.ai/) · [index for agents (`llms.txt`)](https://docs.typesafe.ai/llms.txt)
- [System One](https://docs.typesafe.ai/concepts/system-one) · [How to build with System One](https://docs.typesafe.ai/concepts/how-to-build-with-system-one) · [Use-case map](https://docs.typesafe.ai/concepts/use-case-map)
- Primitives: [Noul](https://docs.typesafe.ai/primitives/noul) · [Choice](https://docs.typesafe.ai/primitives/choice) · [Score](https://docs.typesafe.ai/primitives/score) · [State](https://docs.typesafe.ai/concepts/state) · [Confidence](https://docs.typesafe.ai/confidence)
- Patterns: [speculative fan-out](https://docs.typesafe.ai/patterns/fan-out) · [composite scoring](https://docs.typesafe.ai/patterns/composite-scoring) · [function calling cookbook](https://docs.typesafe.ai/cookbooks/function_calling)
- [Models](https://docs.typesafe.ai/models) · [HTTP API](https://docs.typesafe.ai/api) · [Python SDK](https://docs.typesafe.ai/sdk/python) · [Console / API keys](https://console.typesafe.ai/settings/keys)
- [Agent skills](https://github.com/typesafe-ai/skills)

LangChain
- [TypeSafe integration](https://docs.langchain.com/oss/python/integrations/providers/typesafe) · [`langchain-typesafe` API reference](https://reference.langchain.com/python/langchain-typesafe) · [PyPI](https://pypi.org/project/langchain-typesafe/)
- Middleware: [overview](https://docs.langchain.com/oss/python/langchain/middleware/overview) · [custom](https://docs.langchain.com/oss/python/langchain/middleware/custom) · [built-in, incl. human-in-the-loop](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
- [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) · [LangSmith](https://docs.langchain.com/langsmith/home)

## Status and caveats

- Measured on `jev-1.13.0`, 2026-09-19, with synthetic cases. The rubrics and criteria
  have not been evaluated on a large labelled set; treat 0.5 / 0.6 / 0.8 as starting
  points and re-run the probes after every wording change.
- `langchain-typesafe` is alpha and `TypeSafeClassifier` is marked beta; its API may move.
- The triage recognises Sentry-style short ids (`PROJECT-5G`); other trackers need
  another `_ISSUE_ID`.
- The middlewares target Deep Agents' tool names (`execute`, `task`); `GATED_TOOLS` and
  the `role`/`roles` arguments are where to adapt them.

## License

<!-- TODO before publishing: choose a license and add LICENSE. skills/typesafe-ai keeps its own MIT license. -->
