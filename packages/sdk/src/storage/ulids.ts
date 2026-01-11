import { monotonicFactory } from "ulid";

const monotonicUlid = monotonicFactory();
let lastUlid: string | null = null;

export function nextUlid() {
  lastUlid = monotonicUlid();
  return lastUlid;
}
