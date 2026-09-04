// IndexedDB persistence via Dexie. Conversations carry their images inline,
// so the 5 MB localStorage ceiling the old UI fought against no longer
// applies. A one-time migration lifts the old localStorage history across.
import Dexie, { type Table } from "dexie";
import type { Conversation, Preset } from "../types";

class FlyweightDatabase extends Dexie {
  conversations!: Table<Conversation, string>;
  presets!: Table<Preset, string>;

  constructor() {
    super("flyweight-chat");
    this.version(1).stores({
      conversations: "id, updatedAt, pinned",
      presets: "id, name",
    });
  }
}

export const db = new FlyweightDatabase();

const LEGACY_KEY = "flyweight.chat.v1";

interface LegacyMessage {
  id?: string;
  role: string;
  content: string;
  reasoning?: string;
  images?: string[];
  createdAt?: number;
  toolCalls?: Array<{ id: string; name: string; arguments: string }>;
}

interface LegacyConversation {
  id?: string;
  title?: string;
  createdAt?: number;
  updatedAt?: number;
  messages?: LegacyMessage[];
}

/** Import the old localStorage history once, then leave a marker. */
export async function migrateLegacyHistory(): Promise<number> {
  let raw: string | null = null;
  try {
    raw = localStorage.getItem(LEGACY_KEY);
  } catch {
    return 0;
  }
  if (!raw) return 0;
  let parsed: LegacyConversation[];
  try {
    parsed = JSON.parse(raw);
  } catch {
    return 0;
  }
  if (!Array.isArray(parsed)) return 0;
  const existing = await db.conversations.count();
  if (existing > 0) {
    // Already migrated (or the user has new data); do not merge twice.
    try {
      localStorage.removeItem(LEGACY_KEY);
    } catch {
      /* ignore */
    }
    return 0;
  }
  const now = Date.now();
  const conversations: Conversation[] = parsed
    .filter((item) => Array.isArray(item.messages) && item.messages.length)
    .map((item, index) => ({
      id: item.id ?? `legacy-${now}-${index}`,
      title: item.title ?? "Imported conversation",
      createdAt: item.createdAt ?? now,
      updatedAt: item.updatedAt ?? now,
      messages: (item.messages ?? []).map((message, position) => ({
        id: message.id ?? `legacy-${now}-${index}-${position}`,
        role: (["system", "user", "assistant", "tool"].includes(message.role) ? message.role : "user") as Conversation["messages"][number]["role"],
        content: message.content ?? "",
        reasoning: message.reasoning,
        images: message.images,
        toolCalls: message.toolCalls,
        createdAt: message.createdAt ?? now,
      })),
    }));
  if (conversations.length) await db.conversations.bulkPut(conversations);
  try {
    localStorage.removeItem(LEGACY_KEY);
  } catch {
    /* ignore */
  }
  return conversations.length;
}
