import { beforeEach, describe, expect, it, vi } from "vitest";

const calls = { props: 0 };
let loadedAt = 100;
let workspace: string | undefined;

vi.mock("./lib/api", () => ({
  ApiError: class ApiError extends Error {
    status = 0;
  },
  api: {
    health: vi.fn(async () => ({ status: "ok", model: "m", loaded_at: loadedAt, busy: false })),
    props: vi.fn(async () => {
      calls.props += 1;
      return { model_path: "m", agent_workspace: workspace };
    }),
    models: vi.fn(async () => [{ id: "m" }]),
    slots: vi.fn(async () => []),
  },
}));
vi.mock("./lib/db", () => ({
  db: { listConversations: vi.fn(async () => []), putConversation: vi.fn(), deleteConversation: vi.fn(), listPresets: vi.fn(async () => []) },
  migrateLegacyHistory: vi.fn(async () => undefined),
}));

import { useStore } from "./store";

describe("pollRuntime", () => {
  beforeEach(() => {
    calls.props = 0;
    loadedAt = 100;
    workspace = undefined;
    useStore.setState({ props: null, health: null, models: [] });
  });

  it("reads /props once while the same server keeps answering", async () => {
    await useStore.getState().pollRuntime();
    await useStore.getState().pollRuntime();
    expect(calls.props).toBe(1);
    expect(useStore.getState().props?.agent_workspace).toBeUndefined();
  });

  it("reads /props again after the server restarts, so a new workspace is seen", async () => {
    await useStore.getState().pollRuntime();
    loadedAt = 200;
    workspace = "/work";
    await useStore.getState().pollRuntime();
    expect(calls.props).toBe(2);
    expect(useStore.getState().props?.agent_workspace).toBe("/work");
  });
});
