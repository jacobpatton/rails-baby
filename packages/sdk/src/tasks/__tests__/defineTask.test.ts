import { beforeEach, describe, expect, it } from "vitest";
import {
  defineTask,
  resetGlobalTaskRegistry,
  DuplicateTaskIdError,
  TaskBuildContext,
  globalTaskRegistry,
} from "../../tasks";

let ctxCounter = 0;
const fakeCtx = (overrides: Partial<TaskBuildContext> = {}): TaskBuildContext => {
  const suffix = `${++ctxCounter}`;
  const labels: string[] = overrides.labels ?? [];
  return {
    effectId: overrides.effectId ?? `effect-${suffix}`,
    invocationKey: overrides.invocationKey ?? `invocation-${suffix}`,
    taskId: overrides.taskId ?? `task-${suffix}`,
    runId: overrides.runId ?? "run-1",
    runDir: overrides.runDir ?? "/runs/run-1",
    taskDir: overrides.taskDir ?? `/runs/run-1/tasks/effect-${suffix}`,
    tasksDir: overrides.tasksDir ?? "/runs/run-1/tasks",
    labels,
    createBlobRef:
      overrides.createBlobRef ??
      (async (..._args: Parameters<TaskBuildContext["createBlobRef"]>) => {
        return "blob";
      }),
    toTaskRelativePath: overrides.toTaskRelativePath ?? ((relativePath: string) => relativePath),
  };
};

describe("defineTask id normalization", () => {
  beforeEach(() => {
    resetGlobalTaskRegistry();
  });

  it("trims leading and trailing whitespace from ids", () => {
    const defined = defineTask("  build  ", () => ({ kind: "node" }));
    expect(defined.id).toBe("build");
  });

  it("rejects blank ids", () => {
    expect(() => defineTask("   ", () => ({ kind: "node" }))).toThrowError(
      /defineTask requires a non-empty string id/
    );
  });

  it("preserves already-normalized ids", () => {
    const defined = defineTask("deploy", () => ({ kind: "node" }));
    expect(defined.id).toBe("deploy");
  });
});

describe("defineTask duplicate detection", () => {
  beforeEach(() => {
    resetGlobalTaskRegistry();
  });

  it("throws DuplicateTaskIdError when registering the same id twice", () => {
    defineTask("bundle", () => ({ kind: "node" }));
    expect(() => defineTask("bundle", () => ({ kind: "node" }))).toThrowError(DuplicateTaskIdError);
  });
});

describe("defineTask deterministic output", () => {
  beforeEach(() => {
    resetGlobalTaskRegistry();
  });

  it("returns stable TaskDef instances with cloned labels", async () => {
    const defined = defineTask("deterministic", (_, ctx) => {
      ctx.labels.push("ctx");
      return { kind: "node", labels: ctx.labels };
    });

    const firstCtx = fakeCtx();
    const secondCtx = fakeCtx();
    const first = await defined.build({ value: 1 }, firstCtx);
    const second = await defined.build({ value: 1 }, secondCtx);

    expect(first).toEqual(second);
    expect(first.labels).toEqual(firstCtx.labels);
    expect(second.labels).toEqual(secondCtx.labels);
    expect(first.labels).not.toBe(firstCtx.labels);
    expect(second.labels).not.toBe(secondCtx.labels);
  });
});

describe("defineTask label metadata", () => {
  beforeEach(() => {
    resetGlobalTaskRegistry();
  });

  it("merges option and implementation labels when recording definitions", async () => {
    const defined = defineTask(
      "label-merge",
      () => {
        return { kind: "node", labels: ["impl"] };
      },
      { labels: ["opt"] }
    );

    const ctx = fakeCtx();
    ctx.labels.push("ctx");
    await defined.build({}, ctx);

    const record = globalTaskRegistry.listDefinitions().find((entry) => entry.id === defined.id);
    expect(record?.labels).toEqual(["opt", "impl"]);
  });
});
