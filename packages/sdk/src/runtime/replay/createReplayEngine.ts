import { readRunMetadata, readRunInputs } from "../../storage/runFiles";
import { RunMetadata } from "../../storage/types";
import { buildEffectIndex, EffectIndex } from "./effectIndex";
import { ReplayCursor } from "./replayCursor";
import { ProcessContext } from "../types";
import { createProcessContext, InternalProcessContext } from "../processContext";
import { replaySchemaVersion } from "../constants";
import { RunFailedError } from "../exceptions";
import {
  createStateCacheSnapshot,
  journalHeadsEqual,
  readStateCache,
  StateCacheSnapshot,
  writeStateCache,
} from "./stateCache";

export interface CreateReplayEngineOptions {
  runDir: string;
  now?: () => Date;
  logger?: (...args: any[]) => void;
}

export interface ReplayEngine {
  runId: string;
  runDir: string;
  metadata: RunMetadata;
  inputs?: unknown;
  effectIndex: EffectIndex;
  replayCursor: ReplayCursor;
  context: ProcessContext;
  internalContext: InternalProcessContext;
  stateCache?: StateCacheSnapshot | null;
  stateRebuild?: { reason: string; previous?: { seq: number; ulid: string } | null } | null;
}

export async function createReplayEngine(options: CreateReplayEngineOptions): Promise<ReplayEngine> {
  const metadata = await readRunMetadata(options.runDir);
  ensureCompatibleLayout(metadata.layoutVersion, options.runDir);
  const inputs = await readRunInputs(options.runDir);
  const existingStateCache = await readStateCache(options.runDir);
  const effectIndex = await buildEffectIndex({ runDir: options.runDir });
  const replayCursor = new ReplayCursor();
  const processId = metadata.processId ?? metadata.request ?? metadata.runId;
  const { context, internalContext } = createProcessContext({
    runId: metadata.runId,
    runDir: options.runDir,
    processId,
    effectIndex,
    replayCursor,
    now: options.now,
    logger: options.logger,
  });
  const journalHead = effectIndex.getJournalHead();
  const nextSnapshot = createStateCacheSnapshot(journalHead ?? undefined);
  let stateCacheSnapshot: StateCacheSnapshot | null = existingStateCache ?? null;
  let stateRebuild: ReplayEngine["stateRebuild"] = null;
  if (!existingStateCache) {
    stateCacheSnapshot = nextSnapshot;
    stateRebuild = { reason: "missing_cache", previous: null };
    await writeStateCache(options.runDir, stateCacheSnapshot);
  } else if (!journalHeadsEqual(existingStateCache.journalHead, journalHead ?? null)) {
    stateCacheSnapshot = nextSnapshot;
    stateRebuild = {
      reason: "journal_mismatch",
      previous: existingStateCache.journalHead ?? null,
    };
    await writeStateCache(options.runDir, stateCacheSnapshot);
  } else {
    stateCacheSnapshot = existingStateCache;
  }

  return {
    runId: metadata.runId,
    runDir: options.runDir,
    metadata,
    inputs,
    effectIndex,
    replayCursor,
    context,
    internalContext,
    stateCache: stateCacheSnapshot,
    stateRebuild,
  };
}

function ensureCompatibleLayout(layoutVersion: string | undefined, runDir: string) {
  if (!layoutVersion) {
    throw new RunFailedError("Run metadata is missing layoutVersion", { runDir });
  }
  if (layoutVersion !== replaySchemaVersion) {
    throw new RunFailedError("Run layout version is not supported by this runtime", {
      expected: replaySchemaVersion,
      actual: layoutVersion,
      runDir,
    });
  }
}
