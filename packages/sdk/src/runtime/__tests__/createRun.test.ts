import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import path from "path";
import os from "os";
import { promises as fs } from "fs";
import { createRun } from "../createRun";
import { loadJournal } from "../../storage/journal";
import { readRunMetadata, readRunInputs } from "../../storage/runFiles";
import { DEFAULT_LAYOUT_VERSION } from "../../storage/paths";
import * as ulids from "../../storage/ulids";

let tmpRoot: string;

beforeEach(async () => {
  tmpRoot = await fs.mkdtemp(path.join(os.tmpdir(), "sdk-create-run-"));
});

afterEach(async () => {
  vi.restoreAllMocks();
  await fs.rm(tmpRoot, { recursive: true, force: true });
});

describe("createRun", () => {
  test("generates a run id, persists metadata, and appends RUN_CREATED", async () => {
    vi.spyOn(ulids, "nextUlid").mockReturnValue("01HZWTESTRUNID");
    const entryFile = path.join(tmpRoot, "processes", "pipeline.mjs");
    await fs.mkdir(path.dirname(entryFile), { recursive: true });
    await fs.writeFile(entryFile, "export async function handler() { return 'ok'; }");

    const result = await createRun({
      runsDir: tmpRoot,
      request: "ci/request-001",
      process: {
        processId: "ci/pipeline",
        importPath: entryFile,
        exportName: "handler",
      },
    });

    expect(result.runId).toBe("01HZWTESTRUNID");
    expect(result.runDir).toBe(path.join(tmpRoot, "01HZWTESTRUNID"));

    const metadata = await readRunMetadata(result.runDir);
    expect(metadata.runId).toBe("01HZWTESTRUNID");
    expect(metadata.processId).toBe("ci/pipeline");
    expect(metadata.request).toBe("ci/request-001");
    expect(metadata.entrypoint).toEqual({
      importPath: "../processes/pipeline.mjs",
      exportName: "handler",
    });
    expect(typeof metadata.createdAt).toBe("string");

    const journal = await loadJournal(result.runDir);
    expect(journal).toHaveLength(1);
    expect(journal[0].type).toBe("RUN_CREATED");
    expect(journal[0].data).toMatchObject({
      runId: "01HZWTESTRUNID",
      processId: "ci/pipeline",
      entrypoint: {
        importPath: "../processes/pipeline.mjs",
        exportName: "handler",
      },
    });
    expect(journal[0].data.inputsRef).toBeUndefined();
  });

  test("writes inputs.json and references it from the RUN_CREATED event", async () => {
    const entryFile = path.join(tmpRoot, "process.mjs");
    await fs.writeFile(entryFile, "export async function process() {}");

    const result = await createRun({
      runsDir: tmpRoot,
      process: {
        processId: "demo/process",
        importPath: entryFile,
      },
      inputs: { branch: "main", sha: "abc123" },
    });

    const metadata = await readRunMetadata(result.runDir);
    expect(metadata.entrypoint.exportName).toBe("process");

    const inputs = await readRunInputs(result.runDir);
    expect(inputs).toEqual({ branch: "main", sha: "abc123" });

    const journal = await loadJournal(result.runDir);
    expect(journal[0].data).toMatchObject({
      inputsRef: "inputs.json",
    });
  });
});
