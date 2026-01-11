import { readTaskDefinition } from "../storage/tasks";
import { TaskDef } from "../tasks/types";
import { hydrateCliNodeTaskEnv } from "../runner/env";
import { RunNodeTaskOptions, RunNodeTaskResult, commitNodeResult, runNodeTask } from "../runner/nodeRunner";
import { CommitEffectResultArtifacts, ProcessLogger } from "../runtime/types";

export interface CliRunNodeTaskOptions extends Omit<RunNodeTaskOptions, "task" | "hydration" | "baseEnv"> {
  task?: TaskDef;
  baseEnv?: NodeJS.ProcessEnv;
  invocationKey?: string;
  logger?: ProcessLogger;
}

export interface CliRunNodeTaskResult extends RunNodeTaskResult {
  hydratedKeys: string[];
  missingKeys: string[];
  committed?: CommitEffectResultArtifacts;
}

export async function runNodeTaskFromCli(options: CliRunNodeTaskOptions): Promise<CliRunNodeTaskResult> {
  const task = options.task ?? (await loadTaskDefinition(options.runDir, options.effectId));
  const hydration = hydrateCliNodeTaskEnv(task, {
    cleanEnv: options.cleanEnv,
    envOverrides: options.envOverrides,
    baseEnv: options.baseEnv ?? process.env,
  });
  const result = await runNodeTask({
    ...options,
    task,
    hydration,
  });
  let committed: CommitEffectResultArtifacts | undefined;
  if (!options.dryRun) {
    committed = await commitNodeResult({
      runDir: options.runDir,
      effectId: options.effectId,
      invocationKey: options.invocationKey ?? extractInvocationKey(task),
      logger: options.logger,
      result,
    });
  }
  return {
    ...result,
    hydratedKeys: hydration.hydratedKeys,
    missingKeys: hydration.missingKeys,
    committed,
  };
}

async function loadTaskDefinition(runDir: string, effectId: string): Promise<TaskDef> {
  const def = await readTaskDefinition(runDir, effectId);
  if (!def) {
    throw new Error(`Task definition for effect ${effectId} is missing`);
  }
  return def as TaskDef;
}

function extractInvocationKey(task: TaskDef): string | undefined {
  const raw = (task as Record<string, unknown>)?.invocationKey;
  return typeof raw === "string" && raw.trim().length > 0 ? raw : undefined;
}
