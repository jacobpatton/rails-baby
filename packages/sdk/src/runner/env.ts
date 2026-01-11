import { JsonRecord } from "../storage/types";
import { TaskDef } from "../tasks/types";

export interface NodeEnvHydrationOptions {
  /**
   * Env map already present on the TaskDef (safe, non-secret values).
   */
  nodeEnv?: Record<string, string | undefined>;
  /**
   * Task metadata containing the `redactedEnvKeys` array emitted by the node helper.
   */
  metadata?: JsonRecord;
  /**
   * Source environment to inherit values from (defaults to `process.env`).
   */
  baseEnv?: NodeJS.ProcessEnv;
  /**
   * Explicit overrides (e.g., CLI --env) that should win over `baseEnv` for hydrated keys.
   */
  overrides?: Record<string, string | undefined>;
  /**
   * When false, the returned env starts empty (aside from sanitized + hydrated keys).
   * Redacted keys may still be sourced from `baseEnv`/`overrides` even if inheritance is disabled.
   */
  inheritProcessEnv?: boolean;
}

export interface NodeEnvHydrationResult {
  env: Record<string, string>;
  hydratedKeys: string[];
  missingKeys: string[];
}

export interface ResolveNodeTaskEnvOptions extends Omit<NodeEnvHydrationOptions, "nodeEnv" | "metadata"> {
  /**
   * Optional metadata override; defaults to task.metadata.
   */
  metadata?: JsonRecord;
}

export interface CliHydratedEnvOptions {
  /**
   * When true, start from an empty env (aside from overrides and hydrated keys).
   */
  cleanEnv?: boolean;
  /**
   * Additional env pairs provided via CLI flags.
   */
  envOverrides?: Record<string, string | undefined>;
  /**
    * Base env for hydration (defaults to process.env).
    */
  baseEnv?: NodeJS.ProcessEnv;
}

/**
 * Merge the sanitized TaskDef env map with any redacted keys pulled from the caller's environment.
 */
export function hydrateNodeTaskEnv(options: NodeEnvHydrationOptions = {}): NodeEnvHydrationResult {
  const inherit = options.inheritProcessEnv !== false;
  const baseEnv = options.baseEnv ?? process.env;
  const env = inherit ? cloneProcessEnv(baseEnv) : {};

  if (options.nodeEnv) {
    for (const [key, value] of Object.entries(options.nodeEnv)) {
      if (!key || value === undefined || value === null) continue;
      env[key] = String(value);
    }
  }

  const overrides = normalizeOverrides(options.overrides);
  const metadataKeys = extractRedactedEnvKeys(options.metadata);
  const hydratedKeys: string[] = [];
  const missingKeys: string[] = [];

  for (const key of metadataKeys) {
    const overrideValue = overrides[key];
    const sourceValue = overrideValue !== undefined ? overrideValue : baseEnv?.[key];
    if (sourceValue === undefined) {
      delete env[key];
      missingKeys.push(key);
      continue;
    }
    env[key] = sourceValue;
    hydratedKeys.push(key);
  }

  for (const [key, value] of Object.entries(overrides)) {
    if (!key) continue;
    env[key] = value;
  }

  return {
    env,
    hydratedKeys,
    missingKeys,
  };
}

export function resolveNodeTaskEnv(task: TaskDef, options: ResolveNodeTaskEnvOptions = {}): NodeEnvHydrationResult {
  const metadata = options.metadata ?? task.metadata;
  return hydrateNodeTaskEnv({
    nodeEnv: task.node?.env,
    metadata,
    baseEnv: options.baseEnv,
    overrides: options.overrides,
    inheritProcessEnv: options.inheritProcessEnv,
  });
}

export function hydrateCliNodeTaskEnv(task: TaskDef, options: CliHydratedEnvOptions = {}): NodeEnvHydrationResult {
  return resolveNodeTaskEnv(task, {
    baseEnv: options.baseEnv,
    overrides: options.envOverrides,
    inheritProcessEnv: options.cleanEnv ? false : true,
  });
}

export function extractRedactedEnvKeys(metadata?: JsonRecord): string[] {
  if (!metadata) {
    return [];
  }
  const value = metadata["redactedEnvKeys"];
  const raw = Array.isArray(value) ? value : [];
  const seen = new Set<string>();
  const keys: string[] = [];
  for (const entry of raw) {
    if (typeof entry !== "string") continue;
    const key = entry.trim();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    keys.push(key);
  }
  keys.sort((a, b) => {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
  });
  return keys;
}

function cloneProcessEnv(source?: NodeJS.ProcessEnv): Record<string, string> {
  if (!source) {
    return {};
  }
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(source)) {
    if (!key || typeof value !== "string") continue;
    env[key] = value;
  }
  return env;
}

function normalizeOverrides(overrides?: Record<string, string | undefined>): Record<string, string> {
  if (!overrides) {
    return {};
  }
  const result: Record<string, string> = {};
  for (const [key, value] of Object.entries(overrides)) {
    if (!key || value === undefined || value === null) continue;
    result[key] = String(value);
  }
  return result;
}
