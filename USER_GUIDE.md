# Babysitter + `o` User Guide

## 1. What `o` + Babysitter solve

| Need | How the bundle helps |
| --- | --- |
| Repeatable agent runs | `o` wraps your preferred runner (Codex, Claude Code, Gemini, custom) with a repo-local prompt (`.a5c/o.md`). Each request becomes a **run** with formal inputs, `code/main.js`, `journal.jsonl`, and `state.json`. |
| Quality convergence | Every run executes the `act → score → breakpoint` loop. Babysitter shows score feedback and lets you steer at breakpoints instead of shipping the first draft. |
| Auditability & collaboration | The event-sourced `.a5c/runs/<run_id>/` directory contains prompts, work summaries, artifacts, and logs so another teammate can replay or resume the work. Babysitter exposes these from VS Code. |
| Multi-shell parity | `o` is Bash-first but works via WSL2/Git Bash on Windows. Babysitter automates dispatch/resume/awaiting-input so PowerShell/Git Bash users stay in VS Code. |

## 2. Quality-convergence loop

1. **`act()`** – produce work (code changes, notes, docs).
2. **`score()`** – evaluate work against explicit criteria. Feedback/metrics are stored under `work_summaries/`.
3. **Loop** – if `reward_total` is below the threshold, `act()` runs again with the feedback.
4. **`breakpoint()`** – pauses for human approval (plan review, scope change, mitigation decision).

Babysitter highlights the current step, displays the latest score feedback, and lets you send instructions when `o` awaits input.

## 3. Event-sourced run model

```
.a5c/runs/<run_id>/
  inputs.json          # structured goals/constraints
  code/main.js         # process program (act/score/breakpoint flow)
  journal.jsonl        # append-only events (function calls, breakpoints, notes)
  state.json           # derived state pointer
  prompts/*.md         # agent prompts per function call
  work_summaries/*.md  # agent outputs per call
  artifacts/           # run-specific deliverables (reports, JSON, docs)
```

Babysitter’s Run Details webview surfaces these files so you can:

- Inspect prompts and responses without leaving VS Code.
- Download artifacts or copy them into docs/decks.
- Open the run folder in the editor if you need full-text search.

## 4. Core Babysitter workflows

- **Dispatch / Resume / Awaiting Input** – drive new runs, attach to existing ones, or send responses when `o` pauses at a breakpoint (Enter/Esc/text).
- **Runs Tree View** – watch runs refresh live; double-click to open Run Details or jump straight to `journal.jsonl`.
- **Artifact browser** – preview files under `run/artifacts/` and `run/work_summaries/` so you can build case studies or status updates.
- **Prompt Builder** – browse `.a5c/processes/**` to create structured requests (e.g., GTM strategy, enablement plan) with a couple of clicks.

## 5. Quick start (recap)

```bash
# Install or update o + Babysitter scaffold
./o init

# Validate the runner + creds
./o doctor --show-install-hints

# Launch a structured run from VS Code
# Command Palette → "Babysitter: Dispatch Run"
```

Monitor progress inside the **Babysitter Runs** view or the Run Details webview. Artifacts will accumulate under `.a5c/runs/run-YYYYMMDD-HHMMSS/`.

## 6. Support & troubleshooting tips

- If `./o` warns about missing `mktemp`/`bash`, run inside WSL2 (recommended) or Git Bash with `babysitter.o.install.bashPath` pointing to `bash.exe`.
- When Babysitter doesn’t show new events, click **Refresh** or reopen Run Logs; for persistent issues, reattach to the run (`Babysitter: Resume Run`).
- Ask users for the entire run folder when debugging: `state.json`, `journal.jsonl`, `prompts/`, and `work_summaries/` let you replay the issue.

For deeper details (installation, CLI flags, config resolution), see the upstream [`o` USER_GUIDE](https://raw.githubusercontent.com/a5c-ai/o/refs/heads/main/USER_GUIDE.md).
