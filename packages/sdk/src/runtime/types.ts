import type { JsonRecord, RunMetadata } from "../storage/types";
import type { DefinedTask, TaskBuildContext, TaskDef, TaskInvokeOptions } from "../tasks/types";

export type { DefinedTask, TaskBuildContext, TaskDef, TaskInvokeOptions } from "../tasks/types";

export type ProcessLogger = (...args: any[]) => void;

export type EffectStatus = "requested" | "resolved_ok" | "resolved_error";

export interface SerializedEffectError {
  name?: string;
  message?: string;
  stack?: string;
  data?: unknown;
}

export interface EffectRecord {
  effectId: string;
  invocationKey: string;
  invocationHash?: string;
  stepId: string;
  taskId: string;
  status: EffectStatus;
  kind?: string;
  label?: string;
  labels?: string[];
  taskDefRef?: string;
  inputsRef?: string;
  resultRef?: string;
  error?: SerializedEffectError;
  stdoutRef?: string;
  stderrRef?: string;
  requestedAt?: string;
  resolvedAt?: string;
}

export interface EffectSchedulerHints {
  pendingCount?: number;
  parallelGroupId?: string;
  sleepUntilEpochMs?: number;
}

export interface EffectAction {
  effectId: string;
  invocationKey: string;
  kind: string;
  label?: string;
  labels?: string[];
  taskDef: TaskDef;
  taskId?: string;
  stepId?: string;
  taskDefRef?: string;
  inputsRef?: string;
  requestedAt?: string;
  schedulerHints?: EffectSchedulerHints;
}

export interface CreateRunOptions {
  runsDir: string;
  runId?: string;
  process: {
    processId: string;
    importPath: string;
    exportName?: string;
  };
  request?: string;
  inputs?: unknown;
  processRevision?: string;
  layoutVersion?: string;
  metadata?: JsonRecord;
  lockOwner?: string;
}

export interface CreateRunResult {
  runId: string;
  runDir: string;
  metadata: RunMetadata;
}

export interface ParallelHelpers {
  all<T>(thunks: Array<() => T | Promise<T>>): Promise<T[]>;
  map<TItem, TOut>(items: TItem[], fn: (item: TItem) => TOut | Promise<TOut>): Promise<TOut[]>;
}

export interface ProcessContext {
  now(): Date;
  task<TArgs, TResult>(
    task: DefinedTask<TArgs, TResult>,
    args: TArgs,
    options?: TaskInvokeOptions
  ): Promise<TResult>;
  breakpoint<T = unknown>(payload: T, options?: { label?: string }): Promise<void>;
  sleepUntil(target: string | number, options?: { label?: string }): Promise<void>;
  orchestratorTask<TArgs = unknown, TResult = unknown>(
    payload: TArgs,
    options?: { label?: string }
  ): Promise<TResult>;
  parallel: ParallelHelpers;
  log?: ProcessLogger;
}

export interface OrchestrateOptions {
  runDir: string;
  process?: {
    importPath: string;
    exportName?: string;
  };
  inputs?: unknown;
  now?: Date | (() => Date);
  context?: Record<string, unknown>;
  logger?: ProcessLogger;
}

export type IterationResult =
  | { status: "completed"; output: unknown }
  | { status: "waiting"; nextActions: EffectAction[] }
  | { status: "failed"; error: unknown };

export interface CommitEffectResultOptions {
  runDir: string;
  effectId: string;
  invocationKey?: string;
  logger?: ProcessLogger;
  result: {
    status: "ok" | "error";
    value?: unknown;
    error?: unknown;
    stdout?: string;
    stderr?: string;
    stdoutRef?: string;
    stderrRef?: string;
    startedAt?: string;
    finishedAt?: string;
    metadata?: JsonRecord;
  };
}

export interface CommitEffectResultArtifacts {
  resultRef: string;
  stdoutRef?: string;
  stderrRef?: string;
  startedAt?: string;
  finishedAt?: string;
}
