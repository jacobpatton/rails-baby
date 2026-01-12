# Babysitter CLI & SDK Examples

This guide walks through a realistic flow that exercises the `babysitter` CLI and the new deterministic test harness exposed from `@a5c/babysitter-sdk/testing`. The examples assume you are standing in the repo root (or a project that already vendored the CLI + SDK) and that `.a5c/runs` is the default runs directory.

> **Tip:** All CLI paths in this document are rendered with POSIX separators (matching the CLI output convention) even when running on Windows.

---

## 1. Create a run from a process entrypoint

```bash
babysitter run:create \
  --process-id dev/build \
  --entry processes/build/process.mjs#process \
  --inputs examples/inputs/build.json
```

Typical JSON response (`--json`):

```json
{
  "runId": "run-20260112-130455",
  "runDir": ".a5c/runs/run-20260112-130455",
  "process": {
    "processId": "dev/build",
    "entry": "processes/build/process.mjs#process"
  }
}
```

---

## 2. Inspect run status

```bash
babysitter run:status run-20260112-130455 --json
```

```json
{
  "state": "waiting",
  "lastEvent": "RUN_CREATED#0001 2026-01-12T13:04:56.012Z",
  "pendingByKind": {
    "node": 2
  },
  "metadata": {
    "stateVersion": 1,
    "pendingEffectsByKind": {
      "node": 2
    }
  }
}
```

The CLI prints the same summary in human form when `--json` is omitted:

```
[run:status] state=waiting last=RUN_CREATED#0001 2026-01-12T13:04:56.012Z pending[node]=2 pending[total]=2 stateVersion=1
```

---

## 3. Discover pending effects

```bash
babysitter task:list run-20260112-130455 --pending
```

```
[task:list] pending=2
- ef-build-001 [node requested] build workspace (taskId=build.workspaces)
- ef-lint-001 [node requested] lint sources (taskId=lint.sources)
```

The JSON variant highlights the run-relative artifact refs (all `/` even on Windows):

```json
{
  "tasks": [
    {
      "effectId": "ef-build-001",
      "status": "requested",
      "kind": "node",
      "label": "build workspace",
      "taskDefRef": "tasks/ef-build-001/task.json",
      "resultRef": null,
      "stdoutRef": null,
      "stderrRef": null
    }
  ]
}
```

---

## 4. Inspect a specific effect

```bash
babysitter task:show run-20260112-130455 ef-build-001 --json
```

Key fields in the response:

```json
{
  "effect": {
    "effectId": "ef-build-001",
    "taskId": "build.workspaces",
    "status": "requested",
    "stdoutRef": null
  },
  "task": {
    "kind": "node",
    "node": {
      "entry": "build/scripts/build-workspace.mjs",
      "args": ["--workspace", "frontend"]
    }
  },
  "result": null,
  "largeResult": null
}
```

When `result.json` exceeds 1 MiB the CLI prints `result: see tasks/<id>/result.json` instead of dumping the payload.

---

## 5. Dry-run a task execution

```bash
babysitter task:run run-20260112-130455 ef-build-001 --dry-run
```

```
[task:run] dry-run plan {"command":{"binary":"node","args":["tasks/ef-build-001/node_entry.mjs","--workspace","frontend"],"cwd":"."},"io":{"input":"tasks/ef-build-001/input.json","output":"tasks/ef-build-001/result.json","stdout":"tasks/ef-build-001/stdout.log","stderr":"tasks/ef-build-001/stderr.log"}}
[task:run] status=skipped
```

Dry runs log the serialized plan to stderr, keep stdout silent (to preserve pipelines), and exit `0`.

---

## 6. Auto-run pending nodes with safeguards

```bash
babysitter run:continue run-20260112-130455 \
  --auto-node-tasks \
  --auto-node-max 1 \
  --auto-node-label build
```

Sample stderr:

```
[auto-run] ef-build-001 [node] build workspace
[auto-run] reached --auto-node-max=1
[run:continue] status=waiting autoNode=1 pending[total]=1 stateVersion=2
```

JSON payload (when `--json` is supplied) shows what was executed vs. filtered out:

```json
{
  "status": "waiting",
  "autoRun": {
    "executed": [
      { "effectId": "ef-build-001", "kind": "node", "label": "build workspace" }
    ],
    "pending": [
      { "effectId": "ef-lint-001", "kind": "node", "label": "lint sources" }
    ]
  },
  "metadata": {
    "stateVersion": 2,
    "pendingEffectsByKind": { "node": 1 }
  },
  "pending": [
    { "effectId": "ef-lint-001", "kind": "node", "label": "lint sources" }
  ]
}
```

---

## 7. Unit-test a process with the deterministic harness

The SDK now exports `runToCompletionWithFakeRunner` from `@a5c/babysitter-sdk/testing`. Use it to exercise process logic without invoking real node runners:

```ts
import { runToCompletionWithFakeRunner } from "@a5c/babysitter-sdk/testing";
import { createRun } from "@a5c/babysitter-sdk";
import path from "node:path";
import os from "node:os";
import fs from "node:fs/promises";

test("build pipeline converges", async () => {
  const runsDir = await fs.mkdtemp(path.join(os.tmpdir(), "babysitter-tests-"));
  const { runDir } = await createRun({
    runsDir,
    process: {
      processId: "dev/build",
      importPath: "../processes/build/process.mjs",
      exportName: "process",
    },
    inputs: { branch: "main" },
  });

  const result = await runToCompletionWithFakeRunner({
    runDir,
    resolve(action) {
      if (action.kind === "node") {
        return { status: "ok", value: { value: action.taskDef.metadata?.value ?? 0 } };
      }
      return undefined;
    },
  });

  expect(result.status).toBe("completed");
  expect(result.executed).toHaveLength(2);
});
```

* Each fake resolution can provide `stdout`, `stderr`, timestamps, and metadata.
* If your resolver returns `undefined` for an action, the harness leaves it pending and returns `{ status: "waiting", pending: [...] }`.
* Use `maxIterations` (default `100`) to catch runaway loops, and `onIteration(result)` to inspect intermediate states.

---

## 8. Cleaning up run artifacts

All examples above write into `.a5c/runs/<runId>`. After a tutorial or test completes, remove the directory (or move it under `runs/archive/`) to keep your repository tidy:

```bash
rm -rf .a5c/runs/run-20260112-130455
```

---

Need another scenario documented? Open an issue with the desired flow (CLI flags, harness behavior, etc.) and the team will extend this file. For the deeper specification refer to [`sdk.md`](../sdk.md).
