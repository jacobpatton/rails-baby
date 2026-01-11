import { DuplicateTaskIdError, globalTaskRegistry } from "./registry";
import { DefinedTask, TaskBuildContext, TaskDef, TaskImpl } from "./types";

export interface DefineTaskOptions {
  description?: string;
  labels?: string[];
  kind?: string;
  source?: string;
}

export function defineTask<TArgs = unknown, TResult = unknown>(
  id: string,
  impl: TaskImpl<TArgs, TResult>,
  options: DefineTaskOptions = {}
): DefinedTask<TArgs, TResult> {
  const taskId = normalizeTaskId(id);
  registerTaskId(taskId, options);

  const defined: DefinedTask<TArgs, TResult> = {
    id: taskId,
    async build(args: TArgs, ctx: TaskBuildContext): Promise<TaskDef> {
      const taskDef = await Promise.resolve(impl(args, ctx));
      const normalized = normalizeTaskDef(taskDef);
      const mergedLabels = [...(options.labels ?? []), ...(normalized.labels ?? [])];
      globalTaskRegistry.recordDefinitionMetadata(taskId, {
        kind: normalized.kind,
        description: normalized.description ?? options.description,
        labels: mergedLabels,
      });
      return normalized;
    },
  };

  return Object.freeze(defined);
}

function registerTaskId(taskId: string, options: DefineTaskOptions) {
  try {
    globalTaskRegistry.registerDefinition({
      id: taskId,
      kind: options.kind,
      description: options.description,
      labels: options.labels ?? [],
      source: options.source,
    });
  } catch (error) {
    if (error instanceof DuplicateTaskIdError) {
      throw error;
    }
    throw new DuplicateTaskIdError(taskId);
  }
}

function normalizeTaskId(id: string): string {
  if (typeof id !== "string" || !id.trim()) {
    throw new Error("defineTask requires a non-empty string id");
  }
  return id.trim();
}

function normalizeTaskDef(taskDef: TaskDef): TaskDef {
  if (!taskDef || typeof taskDef !== "object") {
    throw new Error("Task implementations must return a TaskDef object");
  }
  const labels = Array.isArray(taskDef.labels)
    ? taskDef.labels.filter((label): label is string => typeof label === "string" && Boolean(label.trim()))
    : undefined;
  if (labels) {
    taskDef.labels = Array.from(new Set(labels.map((label) => label.trim())));
  }
  return taskDef;
}
