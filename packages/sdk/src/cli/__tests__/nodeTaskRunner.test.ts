import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import os from "os";
import path from "path";
import { promises as fs } from "fs";
import { createRunDir } from "../../storage/createRunDir";
import { writeTaskDefinition } from "../../storage/tasks";
import { TaskDef } from "../../tasks/types";

vi.mock("../../runner/nodeRunner", async () => {
  const actual = await vi.importActual<typeof import("../../runner/nodeRunner")>("../../runner/nodeRunner");
  return {
    ...actual,
    commitNodeResult: vi.fn(async () => ({
      resultRef: "tasks/mock-effect/result.json",
      stdoutRef: "tasks/mock-effect/stdout.log",
      stderrRef: "tasks/mock-effect/stderr.log",
    })),
  };
});

import { commitNodeResult } from "../../runner/nodeRunner";
import { runNodeTaskFromCli } from "../nodeTaskRunner";

const FIXTURE_SCRIPT = path.join(__dirname, "../../../test-fixtures/runner/print-env.js");
const POSIX_SCRIPT = FIXTURE_SCRIPT.split(path.sep).join("/");

describe("runNodeTaskFromCli", () => {
  let runsRoot: string;
  let runDir: string;
  let effectId: string;

  beforeEach(async () => {
    commitNodeResultMock.mockClear();
    runsRoot = await fs.mkdtemp(path.join(os.tmpdir(), "cli-node-runner-"));
    effectId = `ef-${Date.now().toString(16)}`;
    const { runDir: dir } = await createRunDir({
      runsRoot,
      runId: `run-${Date.now().toString(16)}`,
      request: "cli-node-runner-test",
      processPath: "./process.js",
    });
    runDir = dir;
  });

  afterEach(async () => {
    await fs.rm(runsRoot, { recursive: true, force: true });
  });

  it("hydrates secrets using CLI overrides", async () => {
    const task = buildTaskDef(effectId);
    await writeTaskDefinition(runDir, effectId, task as unknown as Record<string, unknown>);

    const result = await runNodeTaskFromCli({
      runDir,
      effectId,
      task,
      workspaceRoot: path.resolve("."),
      envOverrides: { SECRET_TOKEN: "override-secret", PUBLIC_FLAG: "from-cli" },
      baseEnv: {},
    });

    expect(result.output).toMatchObject({
      secret: "override-secret",
      publicFlag: "from-cli",
    });
    expect(result.hydratedKeys).toEqual(["SECRET_TOKEN"]);
    expect(result.missingKeys).toEqual([]);
  });

  it("surfaces missingKeys when CLI cannot hydrate secrets", async () => {
    const task = buildTaskDef(effectId);
    await writeTaskDefinition(runDir, effectId, task as unknown as Record<string, unknown>);

    const result = await runNodeTaskFromCli({
      runDir,
      effectId,
      task,
      workspaceRoot: path.resolve("."),
      cleanEnv: true,
      baseEnv: {},
    });

    expect(result.output).toMatchObject({
      secret: null,
    });
    expect(result.missingKeys).toEqual(["SECRET_TOKEN"]);
  });

  it("commits node results and includes artifact refs", async () => {
    const task = buildTaskDef(effectId);
    await writeTaskDefinition(runDir, effectId, task as unknown as Record<string, unknown>);

    const committed = await runNodeTaskFromCli({
      runDir,
      effectId,
      task,
      workspaceRoot: path.resolve("."),
      baseEnv: { SECRET_TOKEN: "cli-secret" },
    });

    expect(commitNodeResultMock).toHaveBeenCalledTimes(1);
    expect(committed.committed).toEqual({
      resultRef: "tasks/mock-effect/result.json",
      stdoutRef: "tasks/mock-effect/stdout.log",
      stderrRef: "tasks/mock-effect/stderr.log",
    });
    const callArgs = commitNodeResultMock.mock.calls[0][0];
    expect(callArgs).toMatchObject({
      runDir,
      effectId,
      invocationKey: `invoke-${effectId}`,
    });
  });

  it("skips commit when dryRun is enabled", async () => {
    commitNodeResultMock.mockClear();
    const task = buildTaskDef(effectId);
    await writeTaskDefinition(runDir, effectId, task as unknown as Record<string, unknown>);

    const result = await runNodeTaskFromCli({
      runDir,
      effectId,
      task,
      workspaceRoot: path.resolve("."),
      baseEnv: {},
      dryRun: true,
    });

    expect(commitNodeResultMock).not.toHaveBeenCalled();
    expect(result.committed).toBeUndefined();
  });
});

function buildTaskDef(effectId: string): TaskDef {
  return {
    kind: "node",
    invocationKey: `invoke-${effectId}`,
    metadata: { redactedEnvKeys: ["SECRET_TOKEN"] },
    node: {
      entry: POSIX_SCRIPT,
    },
    io: {
      outputJsonPath: `tasks/${effectId}/script-output.json`,
    },
  };
}

const commitNodeResultMock = commitNodeResult as unknown as ReturnType<typeof vi.fn>;
