import path from "path";

export const JOURNAL_DIR = "journal";
export const TASKS_DIR = "tasks";
export const BLOBS_DIR = "blobs";
export const STATE_DIR = "state";
export const ORPHANED_DIR = "orphaned";
export const PROCESS_DIR = "process";
export const RUN_METADATA_FILE = "run.json";
export const INPUTS_FILE = "inputs.json";
export const LOCK_FILE = "run.lock";
export const STATE_FILE = "state.json";

export const DEFAULT_LAYOUT_VERSION = "2026.01-storage-preview";

export function getRunDir(runsRoot: string, runId: string): string {
  return path.join(runsRoot, runId);
}

export function getJournalDir(runDir: string): string {
  return path.join(runDir, JOURNAL_DIR);
}

export function getTasksDir(runDir: string): string {
  return path.join(runDir, TASKS_DIR);
}

export function getBlobsDir(runDir: string): string {
  return path.join(runDir, BLOBS_DIR);
}

export function getStateDir(runDir: string): string {
  return path.join(runDir, STATE_DIR);
}

export function getStateFile(runDir: string): string {
  return path.join(getStateDir(runDir), STATE_FILE);
}

export function getLockPath(runDir: string): string {
  return path.join(runDir, LOCK_FILE);
}
