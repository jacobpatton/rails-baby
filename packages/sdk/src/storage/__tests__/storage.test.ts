import { afterEach, beforeEach, describe, expect, test } from "vitest";
import os from "os";
import path from "path";
import { promises as fs } from "fs";
import { createRunDir } from "../../storage/createRunDir";
import { appendEvent, loadJournal } from "../../storage/journal";
import { snapshotState } from "../../storage/snapshotState";
import { storeTaskArtifacts } from "../../storage/storeTaskArtifacts";
import { getDiskUsage, findOrphanedBlobs } from "../../storage/cleanup";
import { acquireRunLock, releaseRunLock } from "../../storage/lock";

let tmpRoot: string;

beforeEach(async () => {
  tmpRoot = await fs.mkdtemp(path.join(os.tmpdir(), "babysitter-storage-"));
});

afterEach(async () => {
  await fs.rm(tmpRoot, { recursive: true, force: true });
});

describe("storage primitives", () => {
  test("createRunDir scaffolds layout and metadata", async () => {
    const { runDir } = await createRunDir({
      runsRoot: tmpRoot,
      runId: "run-1",
      request: "demo",
      processPath: ".a5c/processes/foo.js",
      layoutVersion: "test-layout",
      inputs: { hello: "world" },
    });
    const runJson = JSON.parse(await fs.readFile(path.join(runDir, "run.json"), "utf8"));
    expect(runJson.layoutVersion).toBe("test-layout");
    expect(await fs.stat(path.join(runDir, "journal"))).toBeDefined();
  });

  test("appendEvent writes sequential journal files", async () => {
    const { runDir } = await createRunDir({
      runsRoot: tmpRoot,
      runId: "run-2",
      request: "append",
      processPath: ".a5c/processes/foo.js",
    });
    await appendEvent({ runDir, eventType: "RUN_CREATED", event: { ok: true } });
    await appendEvent({ runDir, eventType: "EFFECT_REQUESTED", event: { effectId: "01" } });
    const journalDir = path.join(runDir, "journal");
    const files = (await fs.readdir(journalDir)).sort();
    expect(files).toHaveLength(2);
    expect(files[0].startsWith("000001")).toBe(true);
    const events = await loadJournal(runDir);
    expect(events[0].type).toBe("RUN_CREATED");
    expect(events[1].type).toBe("EFFECT_REQUESTED");
  });

  test("snapshotState writes rebuildable cache", async () => {
    const { runDir } = await createRunDir({
      runsRoot: tmpRoot,
      runId: "run-3",
      request: "state",
      processPath: ".a5c/processes/foo.js",
    });
    await snapshotState({ runDir, state: { cursor: 123 }, journalHead: { seq: 1, ulid: "X" } });
    const contents = JSON.parse(await fs.readFile(path.join(runDir, "state", "state.json"), "utf8"));
    expect(contents.state.cursor).toBe(123);
    expect(contents.journalHead.seq).toBe(1);
  });

  test("storeTaskArtifacts writes metadata and blobs", async () => {
    const { runDir } = await createRunDir({
      runsRoot: tmpRoot,
      runId: "run-4",
      request: "tasks",
      processPath: ".a5c/processes/foo.js",
    });
    await storeTaskArtifacts({
      runDir,
      effectId: "effect-1",
      task: { kind: "act" },
      result: { ok: true },
      artifacts: [
        { name: "stdout.txt", data: "hello" },
        { name: "large.bin", data: Buffer.alloc(600 * 1024, 1) },
      ],
    });
    const artifactsManifest = JSON.parse(
      await fs.readFile(path.join(runDir, "tasks/effect-1/artifacts.json"), "utf8")
    );
    expect(artifactsManifest).toHaveLength(2);
    const blobEntry = artifactsManifest.find((a: any) => a.storedAt.startsWith("blobs/"));
    expect(blobEntry).toBeDefined();
  });

  test("disk usage + orphan detection", async () => {
    const { runDir } = await createRunDir({
      runsRoot: tmpRoot,
      runId: "run-5",
      request: "usage",
      processPath: ".a5c/processes/foo.js",
    });
    await appendEvent({ runDir, eventType: "RUN_CREATED", event: {} });
    await storeTaskArtifacts({
      runDir,
      effectId: "effect-usage",
      artifacts: [{ name: "stdout.txt", data: "log" }],
    });
    const usage = await getDiskUsage(tmpRoot, "run-5");
    expect(usage.totalBytes).toBeGreaterThan(0);
    const orphaned = await findOrphanedBlobs(tmpRoot, "run-5");
    expect(Array.isArray(orphaned)).toBe(true);
  });

  test("lock acquisition enforces single writer", async () => {
    const { runDir } = await createRunDir({
      runsRoot: tmpRoot,
      runId: "run-6",
      request: "lock",
      processPath: ".a5c/processes/foo.js",
    });
    await acquireRunLock(runDir, "test-owner");
    await expect(acquireRunLock(runDir, "other")).rejects.toThrow(/run.lock already held/);
    await releaseRunLock(runDir);
    await acquireRunLock(runDir, "test-owner-2");
  });
});
