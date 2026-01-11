import path from "path";
import { createRunDir } from "../storage/createRunDir";
import { appendEvent } from "../storage/journal";
import { acquireRunLock, releaseRunLock } from "../storage/lock";
import { INPUTS_FILE, getRunDir } from "../storage/paths";
import { RunEntrypointMetadata } from "../storage/types";
import { nextUlid } from "../storage/ulids";
import type { CreateRunOptions, CreateRunResult } from "./types";

export async function createRun(options: CreateRunOptions): Promise<CreateRunResult> {
  const runId = options.runId ?? nextUlid();
  validateRunId(runId);
  const runDir = getRunDir(options.runsDir, runId);
  const normalizedEntrypoint = normalizeEntrypoint(runDir, options.process.importPath, options.process.exportName);
  const requestId = options.request ?? options.process.processId ?? runId;
  const { metadata } = await createRunDir({
    runsRoot: options.runsDir,
    runId,
    request: requestId,
    processId: options.process.processId,
    processRevision: options.processRevision,
    layoutVersion: options.layoutVersion,
    inputs: options.inputs,
    entrypoint: normalizedEntrypoint,
    processPath: normalizedEntrypoint.importPath,
    extraMetadata: options.metadata,
  });

  let lockAcquired = false;
  try {
    await acquireRunLock(runDir, options.lockOwner ?? "runtime:createRun");
    lockAcquired = true;
    const eventPayload: Record<string, unknown> = {
      runId,
      processId: metadata.processId,
      entrypoint: metadata.entrypoint,
    };
    if (metadata.processRevision) {
      eventPayload.processRevision = metadata.processRevision;
    }
    if (options.inputs !== undefined) {
      eventPayload.inputsRef = INPUTS_FILE;
    }
    await appendEvent({
      runDir,
      eventType: "RUN_CREATED",
      event: eventPayload,
    });
  } finally {
    if (lockAcquired) {
      await releaseRunLock(runDir);
    }
  }

  return {
    runId,
    runDir,
    metadata,
  };
}

function validateRunId(runId: string) {
  if (typeof runId !== "string" || runId.trim() === "") {
    throw new Error("runId must be a non-empty string");
  }
}

function normalizeEntrypoint(runDir: string, importPath: string, exportName?: string): RunEntrypointMetadata {
  const entryImport = toRunRelativePosix(runDir, importPath);
  return {
    importPath: entryImport,
    exportName: exportName ?? "process",
  };
}

function toRunRelativePosix(runDir: string, importPath: string): string {
  const relative = path.isAbsolute(importPath) ? path.relative(runDir, importPath) : importPath;
  if (!relative || relative === ".") {
    throw new Error("Entrypoint import path must reference a file");
  }
  return path.posix.normalize(relative.replace(/\\/g, "/"));
}
