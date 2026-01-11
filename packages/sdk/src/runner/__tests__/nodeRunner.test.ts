import { beforeEach, afterEach, describe, expect, it } from "vitest";
import os from "os";
import path from "path";
import { promises as fs } from "fs";
import { createRunDir } from "../../storage/createRunDir";
import { writeTaskDefinition } from "../../storage/tasks";
import { TaskDef } from "../../tasks/types";
import { runNodeTask } from "../nodeRunner";

const RUNNER_FIXTURES_DIR = path.join(__dirname, "../../../test-fixtures/runner");
const PRINT_ENV_SCRIPT = path.join(RUNNER_FIXTURES_DIR, "print-env.js");
const POSIX_SCRIPT = toPosixPath(PRINT_ENV_SCRIPT);
const COPY_INPUTS_SCRIPT = toPosixPath(path.join(RUNNER_FIXTURES_DIR, "copy-inputs-to-output.js"));
const EMIT_LOGS_SCRIPT = toPosixPath(path.join(RUNNER_FIXTURES_DIR, "emit-logs.js"));
const SLOW_LOGGER_SCRIPT = toPosixPath(path.join(RUNNER_FIXTURES_DIR, "slow-logger.js"));

describe("runNodeTask", () => {
  let runsRoot: string;
  let runDir: string;
  let effectId: string;

  beforeEach(async () => {
    runsRoot = await fs.mkdtemp(path.join(os.tmpdir(), "node-runner-"));
    effectId = `ef-${Date.now().toString(16)}`;
    const { runDir: dir } = await createRunDir({
      runsRoot,
      runId: `run-${Date.now().toString(16)}`,
      request: "node-runner-test",
      processPath: "./process.js",
    });
    runDir = dir;
  });

  afterEach(async () => {
    await fs.rm(runsRoot, { recursive: true, force: true });
  });

  it("hydrates redacted env keys before spawning the node task", async () => {
    const task = buildTaskDef(effectId);
    await writeTaskDefinition(runDir, effectId, task as unknown as Record<string, unknown>);

    const baseEnv = {
      SECRET_TOKEN: "super-secret",
      PUBLIC_FLAG: "visible",
    };

    const result = await runNodeTask({
      runDir,
      effectId,
      task,
      workspaceRoot: path.resolve("."),
      baseEnv,
    });

    expect(result.output).toMatchObject({
      secret: "super-secret",
      publicFlag: "visible",
    });
    expect(result.hydrated.hydratedKeys).toEqual(["SECRET_TOKEN"]);
    expect(result.hydrated.missingKeys).toEqual([]);
  });

  it("reports missing redacted keys when they are not present", async () => {
    const task = buildTaskDef(effectId);
    await writeTaskDefinition(runDir, effectId, task as unknown as Record<string, unknown>);

    const result = await runNodeTask({
      runDir,
      effectId,
      task,
      workspaceRoot: path.resolve("."),
      baseEnv: { PUBLIC_FLAG: "visible" },
    });

    expect(result.output).toMatchObject({
      secret: null,
      publicFlag: "visible",
    });
    expect(result.hydrated.hydratedKeys).toEqual([]);
    expect(result.hydrated.missingKeys).toEqual(["SECRET_TOKEN"]);
  });

  it("supports clean env mode", async () => {
    const task = buildTaskDef(effectId);
    await writeTaskDefinition(runDir, effectId, task as unknown as Record<string, unknown>);

    const result = await runNodeTask({
      runDir,
      effectId,
      task,
      workspaceRoot: path.resolve("."),
      baseEnv: { SECRET_TOKEN: "clean-secret" },
      cleanEnv: true,
    });

    expect(result.output).toMatchObject({
      secret: "clean-secret",
    });
    expect(result.hydrated.env.PATH).toBeUndefined();
  });

  it("creates default IO files when io hints are omitted", async () => {
    const inlineInputs = { message: "hello", count: 2 };
    const task: TaskDef = {
      kind: "node",
      node: {
        entry: COPY_INPUTS_SCRIPT,
      },
      inputs: inlineInputs,
    };
    await writeTaskDefinition(runDir, effectId, task as unknown as Record<string, unknown>);

    const result = await runNodeTask({
      runDir,
      effectId,
      task,
      workspaceRoot: path.resolve("."),
      baseEnv: process.env,
    });

    expect(result.io).toEqual({
      inputJsonPath: path.join(runDir, "tasks", effectId, "inputs.json"),
      outputJsonPath: path.join(runDir, "tasks", effectId, "result.json"),
      stdoutPath: path.join(runDir, "tasks", effectId, "stdout.log"),
      stderrPath: path.join(runDir, "tasks", effectId, "stderr.log"),
    });

    const stagedInputs = JSON.parse(await fs.readFile(result.io.inputJsonPath, "utf8"));
    expect(stagedInputs).toEqual(inlineInputs);

    const stdoutStat = await fs.stat(result.io.stdoutPath);
    expect(stdoutStat.isFile()).toBe(true);
    const stderrStat = await fs.stat(result.io.stderrPath);
    expect(stderrStat.isFile()).toBe(true);

    const outputPayload = JSON.parse(await fs.readFile(result.io.outputJsonPath, "utf8"));
    expect(outputPayload.inputValue).toEqual(inlineInputs);
    expect(outputPayload.inputExists).toBe(true);
  });

  it("stages inputs from inputsRef files", async () => {
    const inputsFromRef = { from: "inputsRef", nested: { ok: true } };
    const refRelative = `tasks/${effectId}/inputs-ref.json`;
    const refAbsolute = path.join(runDir, "tasks", effectId, "inputs-ref.json");
    await fs.mkdir(path.dirname(refAbsolute), { recursive: true });
    await fs.writeFile(refAbsolute, JSON.stringify(inputsFromRef), "utf8");

    const task: TaskDef = {
      kind: "node",
      node: {
        entry: COPY_INPUTS_SCRIPT,
      },
      inputsRef: refRelative,
    };
    await writeTaskDefinition(runDir, effectId, task as unknown as Record<string, unknown>);

    const result = await runNodeTask({
      runDir,
      effectId,
      task,
      workspaceRoot: path.resolve("."),
      baseEnv: process.env,
    });

    const stagedInputs = JSON.parse(await fs.readFile(result.io.inputJsonPath, "utf8"));
    expect(stagedInputs).toEqual(inputsFromRef);

    expect(result.output).toMatchObject({
      inputValue: inputsFromRef,
      inputExists: true,
    });
  });

  it("streams stdout/stderr logs to disk", async () => {
    const task: TaskDef = {
      kind: "node",
      node: {
        entry: EMIT_LOGS_SCRIPT,
      },
    };
    await writeTaskDefinition(runDir, effectId, task as unknown as Record<string, unknown>);

    const result = await runNodeTask({
      runDir,
      effectId,
      task,
      workspaceRoot: path.resolve("."),
      baseEnv: process.env,
    });

    const stdoutFile = await fs.readFile(result.io.stdoutPath, "utf8");
    const stderrFile = await fs.readFile(result.io.stderrPath, "utf8");

    expect(result.stdout).toEqual(stdoutFile);
    expect(result.stderr).toEqual(stderrFile);
    expect(result.stdout).toContain("stdout: first line");
    expect(result.stderr).toContain("stderr: only line");

    expect(result.output).toMatchObject({
      stdoutCount: 2,
      stderrCount: 1,
    });
  });

  it("marks timed out runs and flushes partial logs", async () => {
    const task: TaskDef = {
      kind: "node",
      node: {
        entry: SLOW_LOGGER_SCRIPT,
      },
    };
    await writeTaskDefinition(runDir, effectId, task as unknown as Record<string, unknown>);

    const timeoutMs = 75;
    const result = await runNodeTask({
      runDir,
      effectId,
      task,
      workspaceRoot: path.resolve("."),
      baseEnv: process.env,
      timeoutMs,
    });

    expect(result.timedOut).toBe(true);
    expect(result.timeoutMs).toBe(timeoutMs);
    expect(result.durationMs).toBeGreaterThanOrEqual(timeoutMs);

    const stdoutFile = await fs.readFile(result.io.stdoutPath, "utf8");
    const stderrFile = await fs.readFile(result.io.stderrPath, "utf8");
    expect(stdoutFile).toContain("tick-");
    expect(stderrFile).toContain("tock-");
    expect(result.stdout).toContain("tick-");
    expect(result.stderr).toContain("tock-");
  });
});

function buildTaskDef(effectId: string): TaskDef {
  return {
    kind: "node",
    metadata: { redactedEnvKeys: ["SECRET_TOKEN"] },
    node: {
      entry: POSIX_SCRIPT,
    },
    io: {
      outputJsonPath: `tasks/${effectId}/script-output.json`,
    },
  };
}

function toPosixPath(filePath: string): string {
  return filePath.split(path.sep).join("/");
}
