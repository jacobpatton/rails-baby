import { describe, expect, it } from "vitest";
import { extractRedactedEnvKeys, hydrateCliNodeTaskEnv, hydrateNodeTaskEnv, resolveNodeTaskEnv } from "../env";
import { TaskDef } from "../../tasks/types";

describe("hydrateNodeTaskEnv", () => {
  it("hydrates redacted keys from base env", () => {
    const metadata = { redactedEnvKeys: ["CI_SECRET", "NODE_AUTH_TOKEN"] };
    const nodeEnv = {
      PUBLIC_URL: "https://example.test",
      LOG_LEVEL: "info",
    };
    const baseEnv = {
      PATH: "/usr/local/bin",
      NODE_AUTH_TOKEN: "token-123",
      CI_SECRET: "ci-secret",
    };

    const result = hydrateNodeTaskEnv({ metadata, nodeEnv, baseEnv });

    expect(result.env.PUBLIC_URL).toBe("https://example.test");
    expect(result.env.LOG_LEVEL).toBe("info");
    expect(result.env.NODE_AUTH_TOKEN).toBe("token-123");
    expect(result.env.CI_SECRET).toBe("ci-secret");
    expect(result.hydratedKeys).toEqual(["CI_SECRET", "NODE_AUTH_TOKEN"]);
    expect(result.missingKeys).toEqual([]);
  });

  it("prefers overrides over base env when hydrating redacted keys", () => {
    const metadata = { redactedEnvKeys: ["GITHUB_TOKEN"] };
    const baseEnv = { GITHUB_TOKEN: "ghp_base" };
    const overrides = { GITHUB_TOKEN: "ghp_override" };

    const result = hydrateNodeTaskEnv({ metadata, baseEnv, overrides });

    expect(result.env.GITHUB_TOKEN).toBe("ghp_override");
    expect(result.hydratedKeys).toEqual(["GITHUB_TOKEN"]);
    expect(result.missingKeys).toEqual([]);
  });

  it("reports missing redacted keys when no base or overrides available", () => {
    const metadata = { redactedEnvKeys: ["DB_PASSWORD"] };
    const baseEnv = { PATH: "/bin" };

    const result = hydrateNodeTaskEnv({ metadata, baseEnv, inheritProcessEnv: false });

    expect(result.env.DB_PASSWORD).toBeUndefined();
    expect(result.env.PATH).toBeUndefined();
    expect(result.hydratedKeys).toEqual([]);
    expect(result.missingKeys).toEqual(["DB_PASSWORD"]);
  });

  it("hydrates redacted keys from base env even when not inheriting process env", () => {
    const metadata = { redactedEnvKeys: ["SECRET_TOKEN"] };
    const baseEnv = { SECRET_TOKEN: "s3cr3t", FOO: "bar" };

    const result = hydrateNodeTaskEnv({ metadata, baseEnv, inheritProcessEnv: false });

    expect(result.env.SECRET_TOKEN).toBe("s3cr3t");
    expect(result.env.FOO).toBeUndefined();
    expect(result.hydratedKeys).toEqual(["SECRET_TOKEN"]);
    expect(result.missingKeys).toEqual([]);
  });

  it("skips empty or duplicate metadata entries", () => {
    const metadata = { redactedEnvKeys: ["  API_KEY ", "", "API_KEY", null] };
    const baseEnv = { API_KEY: "secret" };

    const result = hydrateNodeTaskEnv({ metadata, baseEnv });

    expect(result.env.API_KEY).toBe("secret");
    expect(result.hydratedKeys).toEqual(["API_KEY"]);
    expect(result.missingKeys).toEqual([]);
  });
});

describe("resolveNodeTaskEnv", () => {
  const baseTask: TaskDef = {
    kind: "node",
    metadata: { redactedEnvKeys: ["API_KEY"] },
    node: {
      entry: "./script.js",
      env: { PUBLIC_URL: "https://example.test" },
    },
  };

  it("derives env from task metadata when options omitted", () => {
    const baseEnv = { API_KEY: "secret", PATH: "/bin" };
    const result = resolveNodeTaskEnv(baseTask, { baseEnv });
    expect(result.env.PUBLIC_URL).toBe("https://example.test");
    expect(result.env.API_KEY).toBe("secret");
    expect(result.hydratedKeys).toEqual(["API_KEY"]);
  });

  it("allows overriding metadata/env via options", () => {
    const altMetadata = { redactedEnvKeys: ["DB_PASSWORD"] };
    const baseEnv = { DB_PASSWORD: "hunter2" };
    const result = resolveNodeTaskEnv(baseTask, { metadata: altMetadata, baseEnv });
    expect(result.env.DB_PASSWORD).toBe("hunter2");
    expect(result.hydratedKeys).toEqual(["DB_PASSWORD"]);
    expect(result.missingKeys).toEqual([]);
  });
});

describe("hydrateCliNodeTaskEnv", () => {
  const task: TaskDef = {
    kind: "node",
    metadata: { redactedEnvKeys: ["TOKEN"] },
    node: { entry: "./script.js", env: { LOG_LEVEL: "info" } },
  };

  it("respects cleanEnv flag", () => {
    const baseEnv = { TOKEN: "cli-token", PATH: "/usr/bin" };
    const result = hydrateCliNodeTaskEnv(task, { cleanEnv: true, baseEnv });
    expect(result.env.PATH).toBeUndefined();
    expect(result.env.TOKEN).toBe("cli-token");
    expect(result.env.LOG_LEVEL).toBe("info");
  });

  it("applies CLI overrides", () => {
    const overrides = { TOKEN: "override", EXTRA: "1" };
    const result = hydrateCliNodeTaskEnv(task, { envOverrides: overrides });
    expect(result.env.TOKEN).toBe("override");
    expect(result.env.EXTRA).toBe("1");
    expect(result.hydratedKeys).toEqual(["TOKEN"]);
  });
});

describe("extractRedactedEnvKeys", () => {
  it("returns sorted unique keys", () => {
    const metadata = { redactedEnvKeys: ["B", "a", "b", "A", ""] };
    expect(extractRedactedEnvKeys(metadata)).toEqual(["A", "B", "a", "b"]);
  });

  it("ignores non-array metadata", () => {
    expect(extractRedactedEnvKeys(undefined)).toEqual([]);
    expect(extractRedactedEnvKeys({ redactedEnvKeys: "token" } as any)).toEqual([]);
  });
});
