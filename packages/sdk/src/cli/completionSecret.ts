import * as crypto from "node:crypto";
import type { RunMetadata } from "../storage/types";

const COMPLETION_SECRET_SALT = "babysitter-completion-secret-v1";

export function deriveCompletionSecret(runId: string): string {
  return crypto.createHash("sha256").update(`${runId}:${COMPLETION_SECRET_SALT}`).digest("hex");
}

export function resolveCompletionSecret(metadata: RunMetadata): string {
  return typeof metadata.completionSecret === "string" ? metadata.completionSecret : deriveCompletionSecret(metadata.runId);
}
