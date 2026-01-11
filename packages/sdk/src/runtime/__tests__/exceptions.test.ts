import { describe, expect, test } from "vitest";
import {
  BabysitterRuntimeError,
  EffectPendingError,
  EffectRequestedError,
  ParallelPendingError,
  RunFailedError,
} from "../exceptions";
import { EffectAction, TaskDef } from "../types";
import { buildParallelBatch } from "../../tasks/batching";

const taskDef: TaskDef = {
  kind: "node",
  title: "demo",
};

const effectAction: EffectAction = {
  effectId: "01EFFECT",
  invocationKey: "proc:S000001:task",
  kind: "node",
  label: "demo-label",
  labels: ["demo-label", "group"],
  stepId: "S000001",
  taskId: "task",
  taskDefRef: "tasks/01EFFECT/task.json",
  inputsRef: "tasks/01EFFECT/inputs.json",
  requestedAt: "2026-01-01T00:00:00Z",
  taskDef,
};

function serializeContractError(error: BabysitterRuntimeError) {
  return JSON.parse(
    JSON.stringify({
      name: error.name,
      message: error.message,
      details: error.details,
    })
  );
}

describe("typed exceptions", () => {
  test("EffectRequestedError contract serialization", () => {
    const error = new EffectRequestedError(effectAction);
    expect(serializeContractError(error)).toMatchInlineSnapshot(`
      {
        "details": {
          "action": {
            "effectId": "01EFFECT",
            "inputsRef": "tasks/01EFFECT/inputs.json",
            "invocationKey": "proc:S000001:task",
            "kind": "node",
            "label": "demo-label",
            "labels": [
              "demo-label",
              "group",
            ],
            "requestedAt": "2026-01-01T00:00:00Z",
            "stepId": "S000001",
            "taskDef": {
              "kind": "node",
              "title": "demo",
            },
            "taskDefRef": "tasks/01EFFECT/task.json",
            "taskId": "task",
          },
        },
        "message": "Effect 01EFFECT requested",
        "name": "EffectRequestedError",
      }
    `);
  });

  test("EffectPendingError contract serialization", () => {
    const error = new EffectPendingError(effectAction);
    expect(serializeContractError(error)).toMatchInlineSnapshot(`
      {
        "details": {
          "action": {
            "effectId": "01EFFECT",
            "inputsRef": "tasks/01EFFECT/inputs.json",
            "invocationKey": "proc:S000001:task",
            "kind": "node",
            "label": "demo-label",
            "labels": [
              "demo-label",
              "group",
            ],
            "requestedAt": "2026-01-01T00:00:00Z",
            "stepId": "S000001",
            "taskDef": {
              "kind": "node",
              "title": "demo",
            },
            "taskDefRef": "tasks/01EFFECT/task.json",
            "taskId": "task",
          },
        },
        "message": "Effect 01EFFECT pending",
        "name": "EffectPendingError",
      }
    `);
  });

  test("ParallelPendingError contract serialization", () => {
    const extraAction: EffectAction = {
      ...effectAction,
      effectId: "01EFFECT2",
      invocationKey: "proc:S000002:task",
      stepId: "S000002",
    };
    const error = new ParallelPendingError(buildParallelBatch([effectAction, extraAction]));
    expect(serializeContractError(error)).toMatchInlineSnapshot(`
      {
        "details": {
          "effects": [
            {
              "effectId": "01EFFECT",
              "inputsRef": "tasks/01EFFECT/inputs.json",
              "invocationKey": "proc:S000001:task",
              "kind": "node",
              "label": "demo-label",
              "labels": [
                "demo-label",
                "group",
              ],
              "requestedAt": "2026-01-01T00:00:00Z",
              "stepId": "S000001",
              "taskDef": {
                "kind": "node",
                "title": "demo",
              },
              "taskDefRef": "tasks/01EFFECT/task.json",
              "taskId": "task",
            },
            {
              "effectId": "01EFFECT2",
              "inputsRef": "tasks/01EFFECT/inputs.json",
              "invocationKey": "proc:S000002:task",
              "kind": "node",
              "label": "demo-label",
              "labels": [
                "demo-label",
                "group",
              ],
              "requestedAt": "2026-01-01T00:00:00Z",
              "stepId": "S000002",
              "taskDef": {
                "kind": "node",
                "title": "demo",
              },
              "taskDefRef": "tasks/01EFFECT/task.json",
              "taskId": "task",
            },
          ],
          "payload": {
            "effects": [
              {
                "effectId": "01EFFECT",
                "inputsRef": "tasks/01EFFECT/inputs.json",
                "invocationKey": "proc:S000001:task",
                "kind": "node",
                "label": "demo-label",
                "labels": [
                  "demo-label",
                  "group",
                ],
                "requestedAt": "2026-01-01T00:00:00Z",
                "stepId": "S000001",
                "taskDefRef": "tasks/01EFFECT/task.json",
                "taskId": "task",
              },
              {
                "effectId": "01EFFECT2",
                "inputsRef": "tasks/01EFFECT/inputs.json",
                "invocationKey": "proc:S000002:task",
                "kind": "node",
                "label": "demo-label",
                "labels": [
                  "demo-label",
                  "group",
                ],
                "requestedAt": "2026-01-01T00:00:00Z",
                "stepId": "S000002",
                "taskDefRef": "tasks/01EFFECT/task.json",
                "taskId": "task",
              },
            ],
          },
        },
        "message": "One or more parallel invocations are pending",
        "name": "ParallelPendingError",
      }
    `);
  });

  test("RunFailedError contract serialization", () => {
    const error = new RunFailedError("boom", { path: "/tmp/run/journal/000001.json" });
    expect(serializeContractError(error)).toMatchInlineSnapshot(`
      {
        "details": {
          "path": "/tmp/run/journal/000001.json",
        },
        "message": "boom",
        "name": "RunFailedError",
      }
    `);
  });
});
