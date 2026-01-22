import { describe, expect, test } from "vitest";
import { buildParallelBatch, summarizeEffectAction, toParallelPendingPayload } from "../batching";
import type { EffectAction } from "../../runtime/types";
import type { TaskDef } from "../types";

const baseTaskDef: TaskDef = {
  kind: "node",
  title: "sample",
  metadata: { priority: "low" },
};

function makeAction(effectId: string, overrides: Partial<EffectAction> = {}): EffectAction {
  return {
    effectId,
    invocationKey: `proc:${effectId}`,
    kind: "node",
    label: `label-${effectId}`,
    labels: [`label-${effectId}`, "shared"],
    taskDef: baseTaskDef,
    taskId: `task-${effectId}`,
    stepId: `STEP-${effectId}`,
    taskDefRef: `tasks/${effectId}/task.json`,
    inputsRef: `tasks/${effectId}/inputs.json`,
    requestedAt: `2026-01-01T00:00:0${effectId.slice(-1)}Z`,
    ...overrides,
  };
}

describe("batching helpers", () => {
  test("buildParallelBatch dedupes by effectId while preserving first occurrence order", () => {
    const actionA = makeAction("01A");
    const actionB = makeAction("01B");
    const actionC = makeAction("01C");

    const batch = buildParallelBatch([actionA, actionB, actionA, actionC, actionB]);

    expect(batch.actions.map((action) => action.effectId)).toEqual(["01A", "01B", "01C"]);
    const groupIds = new Set(batch.actions.map((action) => action.schedulerHints?.parallelGroupId));
    expect(groupIds.size).toBe(1);
    expect(Array.from(groupIds)[0]).toBeDefined();
    expect(batch.summaries.map((summary) => summary.effectId)).toEqual(["01A", "01B", "01C"]);
  });

  test("summarizeEffectAction captures enriched metadata for orchestrators", () => {
    const action = makeAction("01D", {
      labels: ["explicit", "shared"],
    });
    const summary = summarizeEffectAction(action);
    expect(summary).toMatchInlineSnapshot(`
      {
        "effectId": "01D",
        "inputsRef": "tasks/01D/inputs.json",
        "invocationKey": "proc:01D",
        "kind": "node",
        "label": "label-01D",
        "labels": [
          "explicit",
          "shared",
        ],
        "metadata": {
          "priority": "low",
        },
        "requestedAt": "2026-01-01T00:00:0DZ",
        "stepId": "STEP-01D",
        "taskDefRef": "tasks/01D/task.json",
        "taskId": "task-01D",
      }
    `);
  });

  test("toParallelPendingPayload exposes serializable summaries", () => {
    const batch = buildParallelBatch([makeAction("01E"), makeAction("01F")]);
    const payload = toParallelPendingPayload(batch);
    expect(JSON.parse(JSON.stringify(payload))).toEqual({
      effects: payload.effects.map((summary) => ({
        effectId: summary.effectId,
        invocationKey: summary.invocationKey,
        taskId: summary.taskId,
        stepId: summary.stepId,
        kind: summary.kind,
        label: summary.label,
        labels: summary.labels,
        taskDefRef: summary.taskDefRef,
        inputsRef: summary.inputsRef,
        requestedAt: summary.requestedAt,
        metadata: summary.metadata,
      })),
    });
  });
});
