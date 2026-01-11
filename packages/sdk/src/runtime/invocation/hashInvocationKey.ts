import crypto from "crypto";

export interface InvocationKeyComponents {
  processId: string;
  stepId: string;
  taskId: string;
}

export interface InvocationKeyInfo {
  key: string;
  digest: string;
  components: InvocationKeyComponents;
}

export function hashInvocationKey(components: InvocationKeyComponents): InvocationKeyInfo {
  const key = `${components.processId}:${components.stepId}:${components.taskId}`;
  const digest = crypto.createHash("sha256").update(key).digest("hex");
  return { key, digest, components };
}
