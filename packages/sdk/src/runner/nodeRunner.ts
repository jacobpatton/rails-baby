import { spawn } from "child_process";
import { promises as fs, createWriteStream } from "fs";
import path from "path";
import { readTaskDefinition } from "../storage/tasks";
import { JsonRecord } from "../storage/types";
import { commitEffectResult } from "../runtime/commitEffectResult";
import { CommitEffectResultArtifacts, ProcessLogger } from "../runtime/types";
import { TaskDef, TaskIOHints } from "../tasks/types";
import { NodeEnvHydrationResult, resolveNodeTaskEnv } from "./env";

const DEFAULT_TIMEOUT_MS = 15 * 60 * 1000;

export interface RunNodeTaskOptions {
  runDir: string;
  effectId: string;
  task?: TaskDef;
  workspaceRoot?: string;
  nodeBinaryPath?: string;
  envOverrides?: Record<string, string | undefined>;
  cleanEnv?: boolean;
  inheritProcessEnv?: boolean;
  baseEnv?: NodeJS.ProcessEnv;
  hydration?: NodeEnvHydrationResult;
  timeoutMs?: number;
  dryRun?: boolean;
}

export interface RunNodeTaskResult {
  task: TaskDef;
  stdout: string;
  stderr: string;
  exitCode: number | null;
  signal: NodeJS.Signals | null;
  timedOut: boolean;
  startedAt: string;
  finishedAt: string;
  durationMs: number;
  timeoutMs: number;
  output?: unknown;
  io: ResolvedIoPaths;
  hydrated: NodeEnvHydrationResult;
}

export class NodeTaskRunnerError extends Error {
  constructor(message: string, public readonly code: string) {
    super(message);
    this.name = "NodeTaskRunnerError";
  }
}

export async function runNodeTask(options: RunNodeTaskOptions): Promise<RunNodeTaskResult> {
  const task = options.task ?? (await loadNodeTask(options.runDir, options.effectId));
  validateNodeTask(task, options.effectId);

  const hydration =
    options.hydration ??
    resolveNodeTaskEnv(task, {
      baseEnv: options.baseEnv ?? process.env,
      overrides: options.envOverrides,
      inheritProcessEnv: options.cleanEnv ? false : options.inheritProcessEnv,
    });

  const entryPath = resolveWorkspacePath(
    options.workspaceRoot ?? options.runDir,
    task.node?.entry,
    "node.entry",
    options.effectId
  )!;
  const cwdPath = resolveWorkspacePath(
    options.workspaceRoot ?? options.runDir,
    task.node?.cwd,
    "node.cwd",
    options.effectId
  );
  const args = task.node?.args ? [...task.node.args] : [];
  const timeoutMs =
    typeof options.timeoutMs === "number" ? options.timeoutMs : task.node?.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  const envForProcess: Record<string, string> = { ...hydration.env };
  const ioPaths = applyIoEnv(envForProcess, options.runDir, options.effectId, task.io);

  if (options.dryRun) {
    const nowIso = new Date().toISOString();
    return {
      task,
      stdout: "",
      stderr: "",
      exitCode: null,
      signal: null,
      timedOut: false,
      startedAt: nowIso,
      finishedAt: nowIso,
      durationMs: 0,
      timeoutMs,
      io: ioPaths,
      hydrated: hydration,
    };
  }

  await ensureParentDir(ioPaths.inputJsonPath);
  await ensureParentDir(ioPaths.outputJsonPath);
  await ensureParentDir(ioPaths.stdoutPath);
  await ensureParentDir(ioPaths.stderrPath);

  await stageTaskInputs({
    task,
    runDir: options.runDir,
    effectId: options.effectId,
    inputJsonPath: ioPaths.inputJsonPath,
  });

  const startedAtMs = Date.now();
  const startedAtIso = new Date(startedAtMs).toISOString();

  const runResult = await spawnNodeProcess({
    command: options.nodeBinaryPath ?? process.execPath,
    args: [entryPath, ...args],
    cwd: cwdPath ?? options.runDir,
    env: envForProcess,
    timeoutMs,
    stdoutPath: ioPaths.stdoutPath,
    stderrPath: ioPaths.stderrPath,
  });

  const finishedAtMs = Date.now();
  const finishedAtIso = new Date(finishedAtMs).toISOString();

  let parsedOutput: unknown;
  parsedOutput = await readOptionalJson(ioPaths.outputJsonPath);

  return {
    task,
    stdout: runResult.stdout,
    stderr: runResult.stderr,
    exitCode: runResult.exitCode,
    signal: runResult.signal,
    timedOut: runResult.timedOut,
    startedAt: startedAtIso,
    finishedAt: finishedAtIso,
    durationMs: finishedAtMs - startedAtMs,
    timeoutMs,
    output: parsedOutput,
    io: ioPaths,
    hydrated: hydration,
  };
}

export interface CommitNodeResultOptions {
  runDir: string;
  effectId: string;
  invocationKey?: string;
  logger?: ProcessLogger;
  result: RunNodeTaskResult;
}

export async function commitNodeResult(options: CommitNodeResultOptions): Promise<CommitEffectResultArtifacts> {
  const { runDir, effectId, invocationKey, logger, result } = options;
  const stdoutRef = toRunRelativePosix(runDir, result.io.stdoutPath);
  const stderrRef = toRunRelativePosix(runDir, result.io.stderrPath);
  const outputJsonRef = toRunRelativePosix(runDir, result.io.outputJsonPath);
  const status: "ok" | "error" =
    !result.timedOut && (result.exitCode === null || result.exitCode === 0) ? "ok" : "error";

  const metadata: JsonRecord = {
    durationMs: result.durationMs,
    timeoutMs: result.timeoutMs,
    exitCode: result.exitCode,
    signal: result.signal ?? undefined,
    timedOut: result.timedOut,
    outputJsonRef,
  };

  return commitEffectResult({
    runDir,
    effectId,
    invocationKey,
    logger,
    result: {
      status,
      value: status === "ok" ? result.output : undefined,
      error: status === "error" ? buildNodeTaskErrorPayload(result, outputJsonRef) : undefined,
      stdout: result.stdout,
      stderr: result.stderr,
      stdoutRef,
      stderrRef,
      startedAt: result.startedAt,
      finishedAt: result.finishedAt,
      metadata,
    },
  });
}

async function loadNodeTask(runDir: string, effectId: string): Promise<TaskDef> {
  const json = await readTaskDefinition(runDir, effectId);
  if (!json) {
    throw new NodeTaskRunnerError(`Task definition for effect ${effectId} is missing`, "missing_task");
  }
  return json as TaskDef;
}

function validateNodeTask(task: TaskDef, effectId: string) {
  if (task.kind !== "node" || !task.node) {
    throw new NodeTaskRunnerError(`Effect ${effectId} is not a node task`, "invalid_task_kind");
  }
  if (!task.node.entry || typeof task.node.entry !== "string") {
    throw new NodeTaskRunnerError(`Effect ${effectId} node.entry is missing`, "missing_entry");
  }
}

function resolveWorkspacePath(
  baseDir: string,
  value: string | undefined,
  field: string,
  effectId: string
): string | undefined {
  if (!value) {
    return undefined;
  }
  const normalized = value.replace(/\\/g, "/");
  const isAbsolute = path.isAbsolute(normalized) || /^[A-Za-z]:\//.test(normalized);
  const resolved = isAbsolute ? normalized : path.join(baseDir, ...normalized.split("/"));
  if (!resolved) {
    throw new NodeTaskRunnerError(`Effect ${effectId} has an invalid ${field}`, "invalid_path");
  }
  return path.normalize(resolved);
}

function resolveRunRelativePath(runDir: string, relative?: string): string | undefined {
  if (!relative) return undefined;
  if (path.isAbsolute(relative) || /^[A-Za-z]:[\\/]/.test(relative)) {
    return path.normalize(relative);
  }
  const normalized = relative.replace(/\\/g, "/");
  return path.join(runDir, ...normalized.split("/"));
}

export interface ResolvedIoPaths {
  inputJsonPath: string;
  outputJsonPath: string;
  stdoutPath: string;
  stderrPath: string;
}

function applyIoEnv(env: Record<string, string>, runDir: string, effectId: string, io?: TaskIOHints): ResolvedIoPaths {
  const normalizedIo = withDefaultIoHints(effectId, io);
  const inputJsonPath = resolveRunRelativePath(runDir, normalizedIo.inputJsonPath)!;
  const outputJsonPath = resolveRunRelativePath(runDir, normalizedIo.outputJsonPath)!;
  const stdoutPath = resolveRunRelativePath(runDir, normalizedIo.stdoutPath)!;
  const stderrPath = resolveRunRelativePath(runDir, normalizedIo.stderrPath)!;
  env.BABYSITTER_INPUT_JSON = inputJsonPath;
  env.BABYSITTER_OUTPUT_JSON = outputJsonPath;
  env.BABYSITTER_STDOUT_PATH = stdoutPath;
  env.BABYSITTER_STDERR_PATH = stderrPath;
  env.BABYSITTER_EFFECT_ID = effectId;
  return { inputJsonPath, outputJsonPath, stdoutPath, stderrPath };
}

async function readOptionalJson(filePath: string): Promise<unknown | undefined> {
  try {
    const contents = await fs.readFile(filePath, "utf8");
    if (!contents.trim()) {
      return undefined;
    }
    return JSON.parse(contents);
  } catch (error) {
    const err = error as NodeJS.ErrnoException;
    if (err.code === "ENOENT") {
      return undefined;
    }
    throw error;
  }
}

function withDefaultIoHints(effectId: string, io?: TaskIOHints): Required<TaskIOHints> {
  const defaulted = {
    inputJsonPath: `tasks/${effectId}/inputs.json`,
    outputJsonPath: `tasks/${effectId}/result.json`,
    stdoutPath: `tasks/${effectId}/stdout.log`,
    stderrPath: `tasks/${effectId}/stderr.log`,
  };
  return {
    inputJsonPath:
      typeof io?.inputJsonPath === "string" && io.inputJsonPath.trim().length > 0
        ? io.inputJsonPath
        : defaulted.inputJsonPath,
    outputJsonPath:
      typeof io?.outputJsonPath === "string" && io.outputJsonPath.trim().length > 0
        ? io.outputJsonPath
        : defaulted.outputJsonPath,
    stdoutPath:
      typeof io?.stdoutPath === "string" && io.stdoutPath.trim().length > 0 ? io.stdoutPath : defaulted.stdoutPath,
    stderrPath:
      typeof io?.stderrPath === "string" && io.stderrPath.trim().length > 0 ? io.stderrPath : defaulted.stderrPath,
  };
}

interface SpawnNodeOptions {
  command: string;
  args: string[];
  cwd: string;
  env: Record<string, string>;
  timeoutMs: number;
  stdoutPath: string;
  stderrPath: string;
}

async function spawnNodeProcess(
  options: SpawnNodeOptions
): Promise<{ stdout: string; stderr: string; exitCode: number | null; signal: NodeJS.Signals | null; timedOut: boolean }> {
  return new Promise((resolve, reject) => {
    const child = spawn(options.command, options.args, {
      cwd: options.cwd,
      env: options.env,
      stdio: ["ignore", "pipe", "pipe"],
    });

    const stdoutStream = createWriteStream(options.stdoutPath, { flags: "w" });
    const stderrStream = createWriteStream(options.stderrPath, { flags: "w" });
    let streamsClosed = false;

    const finishStreams = () => {
      if (streamsClosed) return;
      streamsClosed = true;
      stdoutStream.end();
      stderrStream.end();
    };

    let settled = false;
    const finishWithError = (error: Error) => {
      if (settled) return;
      settled = true;
      if (timeoutHandle) clearTimeout(timeoutHandle);
      finishStreams();
      reject(error);
    };
    const finishWithSuccess = (payload: {
      stdout: string;
      stderr: string;
      exitCode: number | null;
      signal: NodeJS.Signals | null;
      timedOut: boolean;
    }) => {
      if (settled) return;
      settled = true;
      if (timeoutHandle) clearTimeout(timeoutHandle);
      finishStreams();
      resolve(payload);
    };

    let stdout = "";
    let stderr = "";
    let timedOut = false;

    child.stdout?.setEncoding("utf8");
    child.stdout?.on("data", (chunk) => {
      stdout += chunk;
      stdoutStream.write(chunk);
    });
    child.stderr?.setEncoding("utf8");
    child.stderr?.on("data", (chunk) => {
      stderr += chunk;
      stderrStream.write(chunk);
    });

    stdoutStream.on("error", (error) => {
      child.kill();
      finishWithError(
        new NodeTaskRunnerError(`Failed to write stdout log: ${(error as Error).message}`, "stdout_log_error")
      );
    });
    stderrStream.on("error", (error) => {
      child.kill();
      finishWithError(
        new NodeTaskRunnerError(`Failed to write stderr log: ${(error as Error).message}`, "stderr_log_error")
      );
    });

    const timeoutHandle =
      options.timeoutMs > 0
        ? setTimeout(() => {
            timedOut = true;
            child.kill();
          }, options.timeoutMs)
        : undefined;

    child.on("error", (error) => {
      finishWithError(new NodeTaskRunnerError(`Failed to spawn node task: ${error.message}`, "spawn_error"));
    });

    child.on("close", (code, signal) => {
      finishWithSuccess({ stdout, stderr, exitCode: code, signal, timedOut });
    });
  });
}

async function ensureParentDir(filePath: string) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
}

async function stageTaskInputs(options: { task: TaskDef; runDir: string; effectId: string; inputJsonPath: string }) {
  const inlineInputs = (options.task as { inputs?: unknown }).inputs;
  const hasInlineInputs = inlineInputs !== undefined;
  const maybeInputsRef = (options.task as { inputsRef?: unknown }).inputsRef;
  const inputsRef = typeof maybeInputsRef === "string" ? maybeInputsRef : undefined;

  if (!hasInlineInputs && !inputsRef) {
    return;
  }

  const payload =
    hasInlineInputs && inlineInputs !== undefined
      ? inlineInputs
      : await readInputsRefValue(options.runDir, inputsRef!, options.effectId);

  await fs.writeFile(options.inputJsonPath, JSON.stringify(payload, null, 2) + "\n", "utf8");
}

async function readInputsRefValue(runDir: string, ref: string, effectId: string): Promise<unknown> {
  const resolved = resolveRunRelativePath(runDir, ref);
  if (!resolved) {
    throw new NodeTaskRunnerError(`Effect ${effectId} has an invalid inputsRef path`, "invalid_inputs_ref");
  }
  try {
    const contents = await fs.readFile(resolved, "utf8");
    if (!contents.trim()) {
      throw new NodeTaskRunnerError(`Effect ${effectId} inputsRef file is empty`, "empty_inputs_ref");
    }
    return JSON.parse(contents);
  } catch (error) {
    const err = error as NodeJS.ErrnoException;
    if (err.code === "ENOENT") {
      throw new NodeTaskRunnerError(`Effect ${effectId} inputsRef file is missing`, "missing_inputs_ref");
    }
    throw error;
  }
}

function buildNodeTaskErrorPayload(result: RunNodeTaskResult, outputJsonRef?: string): JsonRecord {
  const base =
    result.exitCode === null
      ? "Node task exited without a code"
      : `Node task exited with code ${result.exitCode}${result.signal ? ` (signal ${result.signal})` : ""}`;
  const message = result.timedOut ? `Node task timed out after ${result.timeoutMs}ms` : base;
  return {
    message,
    exitCode: result.exitCode,
    signal: result.signal ?? undefined,
    timedOut: result.timedOut,
    stdout: result.stdout,
    stderr: result.stderr,
    outputJsonRef,
  };
}

function toRunRelativePosix(runDir: string, absolutePath?: string): string | undefined {
  if (!absolutePath) return undefined;
  const relative = path.relative(runDir, absolutePath);
  if (!relative || relative.startsWith("..")) return undefined;
  return relative.replace(/\\/g, "/");
}
