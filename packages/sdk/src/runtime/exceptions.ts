import { EffectAction, SerializedEffectError } from "./types";
import { toParallelPendingPayload, type ParallelBatch } from "../tasks/batching";

export interface BabysitterErrorDetails {
  [key: string]: unknown;
}

export class BabysitterRuntimeError extends Error {
  readonly details?: BabysitterErrorDetails;

  constructor(name: string, message: string, details?: BabysitterErrorDetails) {
    super(message);
    this.name = name;
    this.details = details;
  }
}

export class BabysitterIntrinsicError extends BabysitterRuntimeError {
  readonly isIntrinsic = true;

  constructor(name: string, message: string, details?: BabysitterErrorDetails) {
    super(name, message, details);
  }
}

export class EffectRequestedError extends BabysitterIntrinsicError {
  constructor(public readonly action: EffectAction) {
    super("EffectRequestedError", `Effect ${action.effectId} requested`, { action });
  }
}

export class EffectPendingError extends BabysitterIntrinsicError {
  constructor(public readonly action: EffectAction) {
    super("EffectPendingError", `Effect ${action.effectId} pending`, { action });
  }
}

export class ParallelPendingError extends BabysitterIntrinsicError {
  readonly effects: EffectAction[];
  constructor(public readonly batch: ParallelBatch) {
    super("ParallelPendingError", "One or more parallel invocations are pending", {
      payload: toParallelPendingPayload(batch),
      effects: batch.actions,
    });
    this.effects = batch.actions;
  }
}

export class InvocationCollisionError extends BabysitterRuntimeError {
  constructor(public readonly invocationKey: string) {
    super(
      "InvocationCollisionError",
      `Invocation key ${invocationKey} is already in use within this run`,
      { invocationKey }
    );
  }
}

export class RunFailedError extends BabysitterRuntimeError {
  constructor(message: string, details?: BabysitterErrorDetails) {
    super("RunFailedError", message, details);
  }
}

export class MissingProcessContextError extends BabysitterRuntimeError {
  constructor() {
    super("MissingProcessContextError", "No active process context found on the current async call stack");
  }
}

export class InvalidTaskDefinitionError extends BabysitterRuntimeError {
  constructor(reason: string) {
    super("InvalidTaskDefinitionError", reason);
  }
}

export class InvalidSleepTargetError extends BabysitterRuntimeError {
  constructor(value: string | number) {
    super("InvalidSleepTargetError", `Invalid sleep target: ${value}`);
  }
}

export function isIntrinsicError(error: unknown): error is BabysitterIntrinsicError {
  return Boolean(error && typeof error === "object" && (error as BabysitterIntrinsicError).isIntrinsic);
}

type ErrorWithData = Error & { data?: unknown };

export function rehydrateSerializedError(data?: SerializedEffectError): Error {
  const name = data?.name ?? "TaskError";
  const message = data?.message ?? "Task failed";
  const err = new Error(message);
  err.name = name;
  if (data?.stack) {
    err.stack = data.stack;
  }
  if (data?.data !== undefined) {
    (err as ErrorWithData).data = data.data;
  }
  return err;
}
