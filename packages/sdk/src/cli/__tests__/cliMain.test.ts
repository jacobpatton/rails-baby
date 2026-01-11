import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MockInstance } from "vitest";
import path from "path";
import { createBabysitterCli } from "../main";
import { runNodeTaskFromCli } from "../nodeTaskRunner";
import type { CliRunNodeTaskResult } from "../nodeTaskRunner";
import { orchestrateIteration } from "../../runtime/orchestrateIteration";
import { readRunMetadata } from "../../storage/runFiles";
import type { TaskDef } from "../../tasks/types";

vi.mock("../nodeTaskRunner", () => ({
  runNodeTaskFromCli: vi.fn(),
}));

vi.mock("../../runtime/orchestrateIteration", () => ({
  orchestrateIteration: vi.fn(),
}));

vi.mock("../../storage/runFiles", () => ({
  readRunMetadata: vi.fn(),
}));

const runNodeTaskFromCliMock = runNodeTaskFromCli as unknown as ReturnType<typeof vi.fn>;
const orchestrateIterationMock = orchestrateIteration as unknown as ReturnType<typeof vi.fn>;
const readRunMetadataMock = readRunMetadata as unknown as ReturnType<typeof vi.fn>;

describe("CLI main entry", () => {
  let logSpy: MockInstance<[message?: any, ...optionalParams: any[]], void>;
  let errorSpy: MockInstance<[message?: any, ...optionalParams: any[]], void>;

  beforeEach(() => {
    vi.clearAllMocks();
    logSpy = vi.spyOn(console, "log").mockImplementation(() => undefined);
    errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);
    readRunMetadataMock.mockResolvedValue(mockRunMetadata());
  });

  afterEach(() => {
    logSpy.mockRestore();
    errorSpy.mockRestore();
  });

  it("executes task:run and prints refs", async () => {
    runNodeTaskFromCliMock.mockResolvedValue(
      mockRunResult({
        committed: {
          resultRef: "tasks/run/result.json",
        },
      })
    );

    const cli = createBabysitterCli();
    const exitCode = await cli.run(["task:run", "runs/demo", "ef-123"]);

    expect(exitCode).toBe(0);
    expect(runNodeTaskFromCliMock).toHaveBeenCalledWith({
      runDir: path.resolve("runs/demo"),
      effectId: "ef-123",
      dryRun: false,
    });
    expect(logSpy).toHaveBeenCalledWith("[task:run] status=ok");
  });

  it("reports waiting actions when auto-node is disabled", async () => {
    orchestrateIterationMock.mockResolvedValue({
      status: "waiting",
      nextActions: [
        {
          effectId: "ef-manual",
          kind: "breakpoint",
          label: "needs approval",
        },
      ],
    });

    const cli = createBabysitterCli();
    const exitCode = await cli.run(["run:continue", "runs/demo"]);

    expect(exitCode).toBe(0);
    expect(logSpy).toHaveBeenCalledWith("[run:continue] status=waiting pending=1");
    expect(logSpy).toHaveBeenCalledWith("- ef-manual [breakpoint] needs approval");
  });

  it("auto-runs node actions until completion", async () => {
    const taskDef: TaskDef = {
      kind: "node",
      node: { entry: "./script.js" },
    };
    orchestrateIterationMock.mockResolvedValueOnce({
      status: "waiting",
      nextActions: [
        {
          effectId: "ef-auto",
          kind: "node",
          label: "auto",
          taskDef,
        },
      ],
    });
    orchestrateIterationMock.mockResolvedValueOnce({
      status: "completed",
      output: { ok: true },
    });
    runNodeTaskFromCliMock.mockResolvedValue(mockRunResult());

    const cli = createBabysitterCli();
    const exitCode = await cli.run(["run:continue", "runs/demo", "--auto-node-tasks"]);

    expect(exitCode).toBe(0);
    expect(orchestrateIterationMock).toHaveBeenCalledTimes(2);
    expect(runNodeTaskFromCliMock).toHaveBeenCalledWith({
      runDir: path.resolve("runs/demo"),
      effectId: "ef-auto",
      task: taskDef,
      invocationKey: undefined,
      dryRun: false,
    });
    expect(logSpy).toHaveBeenCalledWith("[auto-run] ef-auto [node] auto");
    expect(logSpy).toHaveBeenCalledWith("[run:continue] status=completed autoNode=1");
  });

  it("emits JSON summary for waiting runs", async () => {
    orchestrateIterationMock.mockResolvedValue({
      status: "waiting",
      nextActions: [
        {
          effectId: "ef-json",
          kind: "breakpoint",
          label: "manual",
        },
      ],
    });

    const cli = createBabysitterCli();
    const exitCode = await cli.run(["run:continue", "runs/demo", "--json"]);

    expect(exitCode).toBe(0);
    const raw = String(logSpy.mock.calls.at(-1)?.[0] ?? "{}");
    const payload = JSON.parse(raw);
    expect(payload).toMatchObject({
      status: "waiting",
      autoRun: {
        executed: [],
        pending: [{ effectId: "ef-json", kind: "breakpoint", label: "manual" }],
      },
      pending: [{ effectId: "ef-json", kind: "breakpoint", label: "manual" }],
    });
  });

  it("reports errors and exits non-zero", async () => {
    orchestrateIterationMock.mockResolvedValue({
      status: "failed",
      error: { message: "boom" },
    });

    const cli = createBabysitterCli();
    const exitCode = await cli.run(["run:continue", "runs/demo"]);

    expect(exitCode).toBe(1);
    expect(errorSpy).toHaveBeenCalledWith("[run:continue] status=failed");
    expect(errorSpy).toHaveBeenCalledWith({ message: "boom" });
  });

  describe("run:step", () => {
    it("runs one iteration and prints completion output", async () => {
      orchestrateIterationMock.mockResolvedValue({
        status: "completed",
        output: { ok: true },
      });

      const cli = createBabysitterCli();
      const exitCode = await cli.run(["run:step", "runs/demo"]);

      expect(exitCode).toBe(0);
      expect(orchestrateIterationMock).toHaveBeenCalledWith({
        runDir: path.resolve("runs/demo"),
        now: expect.any(Date),
      });
      expect(logSpy).toHaveBeenCalledWith('[run:step] status=completed output={"ok":true}');
    });

    it("prints pending action summaries when waiting", async () => {
      orchestrateIterationMock.mockResolvedValue({
        status: "waiting",
        nextActions: [
          {
            effectId: "ef-manual",
            kind: "breakpoint",
            label: "needs approval",
          },
        ],
      });

      const cli = createBabysitterCli();
      const exitCode = await cli.run(["run:step", "runs/demo"]);

      expect(exitCode).toBe(0);
      expect(logSpy).toHaveBeenCalledWith("[run:step] status=waiting pending=1");
      expect(logSpy).toHaveBeenCalledWith("- ef-manual [breakpoint] needs approval");
    });

    it("prints failures and exits with status 1", async () => {
      const errorPayload = { message: "iteration failed" };
      orchestrateIterationMock.mockResolvedValue({
        status: "failed",
        error: errorPayload,
      });

      const cli = createBabysitterCli();
      const exitCode = await cli.run(["run:step", "runs/demo"]);

      expect(exitCode).toBe(1);
      expect(errorSpy).toHaveBeenCalledWith("[run:step] status=failed");
      expect(errorSpy).toHaveBeenCalledWith(errorPayload);
    });

    it("emits the raw iteration payload in JSON mode", async () => {
      const payload = { status: "failed", error: { message: "boom" } };
      orchestrateIterationMock.mockResolvedValue(payload);

      const cli = createBabysitterCli();
      const exitCode = await cli.run(["run:step", "runs/demo", "--json"]);

      expect(exitCode).toBe(1);
      const raw = String(logSpy.mock.calls.at(-1)?.[0] ?? "{}");
      expect(JSON.parse(raw)).toEqual(payload);
    });

    it("parses --now override into a Date", async () => {
      orchestrateIterationMock.mockResolvedValue({
        status: "completed",
        output: null,
      });
      const nowIso = "2026-01-11T00:00:00.000Z";

      const cli = createBabysitterCli();
      const exitCode = await cli.run(["run:step", "runs/demo", "--now", nowIso]);

      expect(exitCode).toBe(0);
      expect(orchestrateIterationMock).toHaveBeenCalledWith({
        runDir: path.resolve("runs/demo"),
        now: new Date(nowIso),
      });
    });

    it("prints usage when runDir is missing", async () => {
      const cli = createBabysitterCli();
      const exitCode = await cli.run(["run:step"]);

      expect(exitCode).toBe(1);
      expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("Usage:"));
      expect(orchestrateIterationMock).not.toHaveBeenCalled();
    });
  });
});

function mockRunResult(overrides: Partial<CliRunNodeTaskResult> = {}): CliRunNodeTaskResult {
  const now = new Date(0).toISOString();
  const base: CliRunNodeTaskResult = {
    task: { kind: "node", node: { entry: "./script.js" } },
    stdout: "",
    stderr: "",
    exitCode: 0,
    signal: null,
    timedOut: false,
    startedAt: now,
    finishedAt: now,
    durationMs: 0,
    timeoutMs: 0,
    output: undefined,
    io: {
      inputJsonPath: path.resolve("runs/demo", "tasks/mock/input.json"),
      outputJsonPath: path.resolve("runs/demo", "tasks/mock/result.json"),
      stdoutPath: path.resolve("runs/demo", "tasks/mock/stdout.log"),
      stderrPath: path.resolve("runs/demo", "tasks/mock/stderr.log"),
    },
    hydrated: { env: {}, hydratedKeys: [], missingKeys: [] },
    hydratedKeys: [],
    missingKeys: [],
    committed: undefined,
  };
  return { ...base, ...overrides };
}

function mockRunMetadata() {
  return {
    runId: "run-demo",
    request: "req-123",
    processId: "process/demo",
    entrypoint: { importPath: "./process.js", exportName: "process" },
    layoutVersion: "1",
    createdAt: new Date(0).toISOString(),
  };
}
