import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import path from "path";
import os from "os";
import { promises as fs } from "node:fs";
import { createBabysitterCli } from "../main";
import { readRunMetadata } from "../../storage/runFiles";
import { DEFAULT_LAYOUT_VERSION } from "../../storage/paths";
import { appendEvent, loadJournal } from "../../storage/journal";
import { createRunDir } from "../../storage/createRunDir";

describe("babysitter run:create CLI", () => {
  let runsRoot: string;
  let logSpy: ReturnType<typeof vi.spyOn>;
  let errorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(async () => {
    runsRoot = await fs.mkdtemp(path.join(os.tmpdir(), "cli-run-create-"));
    logSpy = vi.spyOn(console, "log").mockImplementation(() => undefined);
    errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);
  });

  afterEach(async () => {
    logSpy.mockRestore();
    errorSpy.mockRestore();
    await fs.rm(runsRoot, { recursive: true, force: true });
  });

  it("creates a run on disk and records metadata plus RUN_CREATED event", async () => {
    const entryFile = await writeEntrypoint("processes/pipeline.mjs", `export async function handler() {\n  return "ok";\n}\n`);
    const inputsPath = path.join(runsRoot, "inputs.json");
    await fs.writeFile(inputsPath, JSON.stringify({ branch: "main", sha: "abc123" }));

    const cli = createBabysitterCli();
    const exitCode = await cli.run([
      "run:create",
      "--runs-dir",
      runsRoot,
      "--process-id",
      "ci/pipeline",
      "--entry",
      `${entryFile}#handler`,
      "--inputs",
      inputsPath,
      "--process-revision",
      "rev-42",
      "--request",
      "deploy/req-7",
    ]);

    expect(exitCode).toBe(0);
    expect(logSpy).toHaveBeenCalledWith(expect.stringContaining("[run:create] runId="));

    const runDir = await expectSingleRunDir();
    const metadata = await readRunMetadata(runDir);
    expect(metadata.processId).toBe("ci/pipeline");
    expect(metadata.request).toBe("deploy/req-7");
    expect(metadata.processRevision).toBe("rev-42");
    expect(metadata.layoutVersion).toBe(DEFAULT_LAYOUT_VERSION);
    expect(metadata.entrypoint).toEqual({
      importPath: toPosixRelative(runDir, entryFile),
      exportName: "handler",
    });

    const journal = await loadJournal(runDir);
    expect(journal).toHaveLength(1);
    expect(journal[0].type).toBe("RUN_CREATED");
    expect(journal[0].data).toMatchObject({
      entrypoint: metadata.entrypoint,
      inputsRef: "inputs.json",
      processRevision: "rev-42",
    });
  });

  it("supports machine-readable --json output", async () => {
    const entryFile = await writeEntrypoint("processes/build.mjs", `export async function process() { return true; }\n`);

    const cli = createBabysitterCli();
    const exitCode = await cli.run([
      "run:create",
      "--runs-dir",
      runsRoot,
      "--process-id",
      "ci/build",
      "--entry",
      `${entryFile}#process`,
      "--json",
    ]);

    expect(exitCode).toBe(0);
    const payload = readLastJsonLine(logSpy);
    const runDir = await expectSingleRunDir();
    const metadata = await readRunMetadata(runDir);

    expect(payload).toMatchObject({
      runId: metadata.runId,
      runDir,
      entry: `${metadata.entrypoint.importPath}#${metadata.entrypoint.exportName}`,
    });
  });

  it("fails fast when required flags are missing", async () => {
    const cli = createBabysitterCli();
    const exitCode = await cli.run(["run:create", "--entry", "./process.mjs"]);

    expect(exitCode).toBe(1);
    expect(errorSpy).toHaveBeenCalledWith("--process-id is required for run:create");
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("babysitter run:create"));
  });

  async function writeEntrypoint(relativePath: string, contents: string) {
    const absolutePath = path.join(runsRoot, relativePath);
    await fs.mkdir(path.dirname(absolutePath), { recursive: true });
    await fs.writeFile(absolutePath, contents, "utf8");
    return absolutePath;
  }

  async function expectSingleRunDir(): Promise<string> {
    const entries = await fs.readdir(runsRoot, { withFileTypes: true });
    const runDir = entries.find((entry) => entry.isDirectory() && /^[0-9A-Za-z]{26}$/.test(entry.name));
    if (!runDir) {
      throw new Error(`Expected run directory inside ${runsRoot}, found: ${entries.map((entry) => entry.name).join(", ")}`);
    }
    return path.join(runsRoot, runDir.name);
  }

  function toPosixRelative(fromDir: string, target: string): string {
    const relative = path.relative(fromDir, target).replace(/\\/g, "/");
    return path.posix.normalize(relative);
  }

  function readLastJsonLine(spy: ReturnType<typeof vi.spyOn>) {
    const raw = String(spy.mock.calls.at(-1)?.[0] ?? "{}");
    return JSON.parse(raw);
  }
});

describe("run lifecycle inspection commands", () => {
  let runsRoot: string;
  let cli: ReturnType<typeof createBabysitterCli>;
  let logSpy: ReturnType<typeof vi.spyOn>;
  let errorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(async () => {
    runsRoot = await fs.mkdtemp(path.join(os.tmpdir(), "cli-run-inspect-"));
    cli = createBabysitterCli();
    logSpy = vi.spyOn(console, "log").mockImplementation(() => undefined);
    errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);
  });

  afterEach(async () => {
    logSpy.mockRestore();
    errorSpy.mockRestore();
    await fs.rm(runsRoot, { recursive: true, force: true });
  });

  describe("run:status", () => {
    it("reports state, last event, and pending counts", async () => {
      const runDir = await createRunWithPendingEffects();

      const exitCode = await cli.run(["run:status", runDir]);

      expect(exitCode).toBe(0);
      const line = findSingleLine(logSpy, (entry) => entry.startsWith("[run:status]"));
      expect(line).toContain("state=waiting");
      expect(line).toContain("last=EFFECT_REQUESTED#000003");
      expect(line).toContain("pending[total]=2");
      expect(line).toContain("pending[breakpoint]=1");
      expect(line).toContain("pending[node]=1");
    });

    it("emits JSON payload with state and last event details", async () => {
      const runDir = await createRunWithPendingEffects();

      const exitCode = await cli.run(["run:status", runDir, "--json"]);

      expect(exitCode).toBe(0);
      const payload = readLastJson(logSpy);
      expect(payload.state).toBe("waiting");
      expect(payload.pendingByKind).toEqual({ breakpoint: 1, node: 1 });
      expect(payload.lastEvent).toMatchObject({ type: "EFFECT_REQUESTED", seq: 3 });
      expect(payload.lastEvent.path).toMatch(/^journal\//);
    });

    it("fails clearly when the run directory is missing", async () => {
      const missingDir = path.join(runsRoot, "missing-run");

      const exitCode = await cli.run(["run:status", missingDir]);

      expect(exitCode).toBe(1);
      expect(hasLineContaining(errorSpy, "[run:status] unable to read run metadata")).toBe(true);
    });

    it("reports created state when no work has been requested yet", async () => {
      const runDir = await createRunSkeleton("run-empty");

      const exitCode = await cli.run(["run:status", runDir]);

      expect(exitCode).toBe(0);
      const line = findSingleLine(logSpy, (entry) => entry.startsWith("[run:status]"));
      expect(line).toContain("state=created");
      expect(line).toContain("pending[total]=0");
    });

    it("honors terminal RUN_* events even if pending work remains", async () => {
      const runDir = await createRunWithHistory();

      const exitCode = await cli.run(["run:status", runDir]);

      expect(exitCode).toBe(0);
      const line = findSingleLine(logSpy, (entry) => entry.startsWith("[run:status]"));
      expect(line).toContain("state=failed");
      expect(line).toContain("pending[breakpoint]=1");
      expect(line).toContain("last=RUN_FAILED#000005");
    });
  });

  describe("run:events", () => {
    it("lists events with filtering, reverse ordering, and limits", async () => {
      const runDir = await createRunWithHistory();

      const exitCode = await cli.run([
        "run:events",
        runDir,
        "--filter-type",
        "effect_requested",
        "--limit",
        "1",
        "--reverse",
      ]);

      expect(exitCode).toBe(0);
      const header = findSingleLine(logSpy, (entry) => entry.startsWith("[run:events]"));
      expect(header).toContain("total=5");
      expect(header).toContain("matching=2");
      expect(header).toContain("showing=1");
      expect(header).toContain("filter=EFFECT_REQUESTED");
      expect(header).toContain("limit=1");
      expect(header).toContain("order=desc");
      const entries = collectPrefixed(logSpy, "- ");
      expect(entries).toHaveLength(1);
      expect(entries[0]).toContain("#000003");
      expect(entries[0]).toContain("EFFECT_REQUESTED");
    });

    it("emits JSON event arrays", async () => {
      const runDir = await createRunWithHistory();

      const exitCode = await cli.run(["run:events", runDir, "--limit", "2", "--json"]);

      expect(exitCode).toBe(0);
      const payload = readLastJson(logSpy);
      expect(Array.isArray(payload.events)).toBe(true);
      expect(payload.events).toHaveLength(2);
      expect(payload.events[0]).toMatchObject({ seq: 1, type: "RUN_CREATED" });
      expect(payload.events[1]).toMatchObject({ seq: 2, type: "EFFECT_REQUESTED" });
    });

    it("fails fast when --limit is invalid", async () => {
      const runDir = await createRunWithHistory();

      const exitCode = await cli.run(["run:events", runDir, "--limit", "0"]);

      expect(exitCode).toBe(1);
      expect(hasLineContaining(errorSpy, "--limit must be a positive integer")).toBe(true);
    });

    it("fails clearly when the run directory cannot be read", async () => {
      const missingDir = path.join(runsRoot, "missing-events");

      const exitCode = await cli.run(["run:events", missingDir]);

      expect(exitCode).toBe(1);
      expect(hasLineContaining(errorSpy, "[run:events] unable to read run metadata")).toBe(true);
    });
  });

  async function createRunWithPendingEffects() {
    const runDir = await createRunSkeleton("run-pending");
    await appendRequestedEffect(runDir, "ef-node", "node", "build");
    await appendRequestedEffect(runDir, "ef-break", "breakpoint", "manual");
    return runDir;
  }

  async function createRunWithHistory() {
    const runDir = await createRunSkeleton("run-history");
    await appendRequestedEffect(runDir, "ef-node", "node", "build");
    await appendRequestedEffect(runDir, "ef-break", "breakpoint", "manual");
    await appendResolvedEffect(runDir, "ef-node");
    await appendEvent({
      runDir,
      eventType: "RUN_FAILED",
      event: { reason: "boom" },
    });
    return runDir;
  }

  async function createRunSkeleton(runId: string) {
    const { runDir } = await createRunDir({
      runsRoot,
      runId,
      request: "cli-test",
      processPath: "./process.js",
    });
    await appendEvent({
      runDir,
      eventType: "RUN_CREATED",
      event: { runId },
    });
    return runDir;
  }

  async function appendRequestedEffect(runDir: string, effectId: string, kind: string, label: string) {
    const refs = await writeTaskFiles(runDir, effectId, kind);
    await appendEvent({
      runDir,
      eventType: "EFFECT_REQUESTED",
      event: {
        effectId,
        invocationKey: `${effectId}:inv`,
        stepId: `step-${effectId}`,
        taskId: `${kind}-task`,
        kind,
        label,
        taskDefRef: refs.taskDefRef,
        inputsRef: refs.inputsRef,
      },
    });
  }

  async function appendResolvedEffect(runDir: string, effectId: string) {
    await appendEvent({
      runDir,
      eventType: "EFFECT_RESOLVED",
      event: {
        effectId,
        status: "ok",
        resultRef: `tasks/${effectId}/result.json`,
        stdoutRef: `tasks/${effectId}/stdout.log`,
        stderrRef: `tasks/${effectId}/stderr.log`,
      },
    });
  }

  async function writeTaskFiles(runDir: string, effectId: string, kind: string) {
    const taskDir = path.join(runDir, "tasks", effectId);
    await fs.mkdir(taskDir, { recursive: true });
    const taskDefPath = path.join(taskDir, "task.json");
    const inputsPath = path.join(taskDir, "inputs.json");
    await fs.writeFile(taskDefPath, JSON.stringify({ kind, schemaVersion: "test" }, null, 2));
    await fs.writeFile(inputsPath, JSON.stringify({ effectId }, null, 2));
    return {
      taskDefRef: `tasks/${effectId}/task.json`,
      inputsRef: `tasks/${effectId}/inputs.json`,
    };
  }

  function readLastJson(spy: ReturnType<typeof vi.spyOn>) {
    const raw = linesFrom(spy).at(-1) ?? "{}";
    return JSON.parse(raw);
  }

  function linesFrom(spy: ReturnType<typeof vi.spyOn>) {
    return spy.mock.calls.map((call) => call.map((value) => String(value ?? "")).join(" "));
  }

  function findSingleLine(spy: ReturnType<typeof vi.spyOn>, predicate: (line: string) => boolean) {
    const line = linesFrom(spy).find(predicate);
    if (!line) {
      throw new Error(`Expected matching log line. Logs:\n${linesFrom(spy).join("\n")}`);
    }
    return line;
  }

  function collectPrefixed(spy: ReturnType<typeof vi.spyOn>, prefix: string) {
    return linesFrom(spy).filter((line) => line.trimStart().startsWith(prefix));
  }

  function hasLineContaining(spy: ReturnType<typeof vi.spyOn>, needle: string) {
    return linesFrom(spy).some((line) => line.includes(needle));
  }
});
