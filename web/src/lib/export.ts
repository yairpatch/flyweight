import type { Conversation } from "../types";
import { formatDate } from "./format";

export function conversationToMarkdown(conversation: Conversation): string {
  const lines: string[] = [`# ${conversation.title}`, "", `_${formatDate(conversation.createdAt)}_`, ""];
  for (const message of conversation.messages) {
    const heading = message.role === "user" ? "User" : message.role === "assistant" ? "Assistant" : message.role === "tool" ? `Tool result (${message.toolName ?? message.toolCallId ?? "tool"})` : "System";
    lines.push(`## ${heading}`, "");
    if (message.reasoning) {
      lines.push("<details><summary>Thinking</summary>", "", message.reasoning, "", "</details>", "");
    }
    if (message.images?.length) lines.push(`_${message.images.length} image${message.images.length === 1 ? "" : "s"} attached_`, "");
    if (message.content) lines.push(message.content, "");
    for (const call of message.toolCalls ?? []) {
      lines.push(`**Tool call** \`${call.name}\``, "", "```json", call.arguments, "```", "");
    }
  }
  return lines.join("\n");
}
