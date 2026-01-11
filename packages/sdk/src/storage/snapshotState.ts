import { SnapshotStateOptions } from "./types";
import { getStateFile } from "./paths";
import { writeFileAtomic } from "./atomic";

export async function snapshotState(options: SnapshotStateOptions) {
  const payload = {
    journalHead: options.journalHead ?? null,
    savedAt: new Date().toISOString(),
    state: options.state,
  };
  await writeFileAtomic(getStateFile(options.runDir), JSON.stringify(payload, null, 2) + "\n");
  return payload;
}
