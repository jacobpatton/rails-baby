# babysitter ide

Babysitter is a VS Code extension for orchestrating and monitoring `o` runs.

## Install

This repository is the extension source. To install a built VSIX:

- Reproducible build + package: `npm ci && npm run package`
- Install the VSIX:
  - VS Code: Extensions view -> `...` -> **Install from VSIX...**
  - Or: `code --install-extension babysitter-*.vsix`

## Develop

- Install deps: `npm install`
- Build once: `npm run build`
- Watch (recommended while iterating): `npm run watch`
- Lint/format: `npm run lint` / `npm run format`
- Tests:
  - All: `npm test`
  - Headless (CI-friendly): `npm run test:ci`

## CI

On pull requests, GitHub Actions runs lint, unit tests, and extension tests, and also packages a VSIX for download/verification. See the workflow run artifacts: `test-logs` and `vsix`.

## Release notes

See `CHANGELOG.md`.

## Run (extension)

- Open this repo in VS Code
- Start the extension host: press `F5`
- Command Palette:
  - `Babysitter: Activate (Log Output)`
  - `Babysitter: Dispatch Run`
  - `Babysitter: Prompt Builder`
  - `Babysitter: Resume Run`
  - `Babysitter: Send ESC to \`o\``
  - `Babysitter: Send Enter to \`o\``
  - `Babysitter: Locate \`o\` Binary`
  - `Babysitter: Install/Update \`o\` in Workspace`
  - `Babysitter: Show Configuration Errors`
  - `Babysitter: Open Run Details`
  - `Babysitter: Open Run Logs`
  - `Babysitter: Reveal Run Folder in Explorer`
  - `Babysitter: Refresh Runs`

## Keyboard shortcuts

Babysitter ships a small set of chorded shortcuts (press the first combo, then the second key):

- `Ctrl+Alt+B`, then `P`: Prompt Builder
- `Ctrl+Alt+B`, then `N`: Dispatch Run
- `Ctrl+Alt+B`, then `R`: Resume Run
- `Ctrl+Alt+B`, then `D`: Open Run Details
- `Ctrl+Alt+B`, then `L`: Open Run Logs
- `Ctrl+Alt+B`, then `Shift+R`: Refresh Runs
- `Ctrl+Alt+B`, then `Enter`: Send Enter to `o`
- `Ctrl+Alt+B`, then `Esc`: Send ESC to `o`

Inside webviews:

- Prompt Builder: `Ctrl+Enter` dispatch, `Ctrl+Shift+Enter` insert, `Ctrl+F` focus search, `Ctrl+L` focus request
- Run Details (when awaiting input): `Ctrl+R` refresh, `Ctrl+Enter` send response, `Enter` send Enter, `Esc` send ESC, `/` focus input
- Run Logs: `Ctrl+F` focus filter, `Ctrl+L` clear active source, `Esc` clears filter

## Runs View

Babysitter adds a **Babysitter Runs** TreeView in the Explorer sidebar showing discovered runs and their current status.

- Click a run (or run `Babysitter: Open Run Details`) to open a Run Details webview with:
  - Key run metadata + current `state.json`
  - Latest `journal.jsonl` events
  - Work summaries (`run/work_summaries/`) with preview + open-in-editor
  - Artifacts browser (`run/artifacts/`) with reveal + open-in-editor
- Open streaming output/logs for a run: `Babysitter: Open Run Logs` (tails `journal.jsonl` and live `o` output when available).
- Right-click a run to reveal its run folder in Explorer/Finder.

## Flows

Babysitter is designed around these primary user flows (per `requirements.md`):

### Dispatch

- Create a new run by dispatching an `o` request.
- Use `Babysitter: Dispatch Run` to invoke `o` with your request prompt.
- For a guided prompt UI that scans `.a5c/processes/` (and supports file drag/drop), use `Babysitter: Prompt Builder`.

### Monitor

- Follow a run by inspecting its run directory, including `journal.jsonl`, `state.json`, and artifacts under `run/artifacts/`.
- Run views auto-refresh on changes to `state.json`, `journal.jsonl`, and `run/artifacts/**` (debounced).
- Parsers are resilient to partial writes (e.g. `journal.jsonl` line currently being written).
- Runs are discovered by scanning the runs root (default `.a5c/runs`) for directories named like `run-YYYYMMDD-HHMMSS`.
- View task stdout and work summaries as the run progresses.

### Resume

- Resume an existing run by re-dispatching using the run id and the updated request/prompt.
- Use `Babysitter: Resume Run` to select a run and enter an updated request, then Babysitter invokes `o` with `[runId, prompt]`.
- Continue monitoring from the same run directory and journal stream.

### Pause

- When `o` is awaiting input (breakpoints/prompts), the Run Details view shows an **Awaiting Input** card with:
  - A text box to send a response/steering instruction
  - Buttons to send `Enter` / `ESC`
- You can also send raw keys from the Command Palette:
  - `Babysitter: Send ESC to \`o\``
  - `Babysitter: Send Enter to \`o\``

Note: Babysitter runs `o` under a pseudo-terminal so interactive key presses work reliably. Output is captured as a single stream (stdout/stderr are not separated when attached to a PTY).

## Prompt Builder / Process Catalog

`Babysitter: Prompt Builder` scans `.a5c/processes/**/*.js` and builds an in-memory catalog of exported process functions.

- Schema: `src/core/processCatalog.ts` (`ProcessCatalog` -> `ProcessExport` -> `ProcessParam`)
- Extraction: parse JS with the TypeScript AST to find exported functions/arrow functions, infer parameter names/defaults/rest, and (best-effort) return-object keys; classify kinds from paths (core/loop/role/recipe/aspect/domain/shared).
- File drag/drop: drag files from the VS Code Explorer into the **Files** drop zone to include them in the generated prompt (workspace files become workspace-relative paths; external files stay as `file://` links).
- Persistence: the Prompt Builder remembers your last selected process, args, request, and files per-workspace.

## Manual QA checklist

- Configuration: verify status bar shows Ready, and `Babysitter: Locate o Binary` finds `o`
- Runs Tree: verify empty state and Refresh Runs behavior
- Dispatch: dispatch a run and confirm it appears in the Runs view
- Prompt Builder: scan processes, select process, generate prompt, drag/drop file link, dispatch
- Run Details: open a run, verify loading/error state, work summary preview, artifacts open/reveal
- Awaiting Input: verify Awaiting Input card shows, send response, send Enter/ESC, and keyboard shortcuts work
- Run Logs: open logs, switch sources, filter, follow, copy/clear

## Troubleshooting

### Babysitter can't find the `o` binary

Babysitter expects `o` to be either:

- In the workspace root (this repo includes an `o` binary at `./o`), or
- Available on `PATH`, or
- Configured explicitly via VS Code setting `babysitter.o.binaryPath` (or legacy `babysitter.oPath`)

If you want Babysitter to install/update `o` into a workspace folder, use `Babysitter: Install/Update \`o\` in Workspace`.

Use `Babysitter: Locate \`o\` Binary` to see what Babysitter is resolving and where it was found.

If you recently changed `PATH`, fully restart VS Code so the extension host picks up the updated environment.

### Installing/updating `o` in a workspace

`Babysitter: Install/Update \`o\` in Workspace` downloads and runs the upstream `o` installer (`curl | bash`) and installs into the selected workspace folder.

- It creates/updates `<workspaceRoot>/o` and `<workspaceRoot>/.a5c/*` (and may edit `<workspaceRoot>/.gitignore` unless you select `--no-gitignore`).
- To update later, rerun the command.
  - Select `--force` to overwrite existing files.
  - Select `--no-gitignore` to avoid managed `.gitignore` edits.
- Windows:
  - WSL2 is recommended (the command will use `wsl.exe` + `bash`).
  - If WSL is unavailable, install Git Bash and set `babysitter.o.install.bashPath` to your `bash.exe` (or let Babysitter auto-detect it).
- Security: the command is gated by a modal consent prompt because it executes a downloaded script.

### Configuration validation

Babysitter validates its configuration on activation and shows a status bar indicator when something is wrong.

- View details: `Babysitter: Show Configuration Errors`
- Settings:
  - `babysitter.o.binaryPath`: path to the `o` executable (or a directory containing it)
  - `babysitter.runsRoot`: runs root (default `.a5c/runs`, relative to the workspace root)
  - `babysitter.globalConfigPath`: optional JSON config file with keys `oBinaryPath` and `runsRoot`
