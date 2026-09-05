import { useEffect, useRef, useState } from "react";
import { ArrowDown, Bot, Sparkles } from "lucide-react";
import { useActiveConversation, useStore } from "../store";
import { MessageItem } from "./MessageItem";

const STICK_THRESHOLD = 8;
const AFFORDANCE_THRESHOLD = 96;

const SUGGESTIONS = [
  "Explain how speculative decoding works in two paragraphs.",
  "Write a Python script that parses a GGUF header and prints its metadata.",
  "Summarize the trade-offs between Q4_K_M and IQ4_XS quantization.",
  "Draft a JSON schema for a weather tool and call it for Tel Aviv.",
];

export function Transcript() {
  const conversation = useActiveConversation();
  const sendMessage = useStore((state) => state.sendMessage);
  const setPanel = useStore((state) => state.setPanel);
  const mode = useStore((state) => state.mode);
  const ready = useStore((state) => state.ready);
  const scroller = useRef<HTMLDivElement>(null);
  const inner = useRef<HTMLDivElement>(null);
  // Whether the view follows the stream. A ref rather than state: the resize
  // observer reads it on every content change, and a user scroll flips it
  // without a re-render.
  const stuck = useRef(true);
  const [showJump, setShowJump] = useState(false);

  const messages = conversation?.messages ?? [];

  const scrollToBottom = (behavior: ScrollBehavior = "auto") => {
    const element = scroller.current;
    if (!element) return;
    element.scrollTo({ top: element.scrollHeight, behavior });
  };

  // Follow the stream: any growth of the content (tokens, a code block
  // finishing, math fonts arriving) scrolls to the bottom while stuck. The
  // observer sees layout changes that a React effect keyed on text length
  // would miss.
  useEffect(() => {
    const element = inner.current;
    if (!element) return;
    const observer = new ResizeObserver(() => {
      if (stuck.current) {
        scrollToBottom();
        return;
      }
      // Content growing below a detached viewport fires no scroll event, so
      // the jump affordance has to be refreshed here.
      const scrollerElement = scroller.current;
      if (scrollerElement) {
        setShowJump(scrollerElement.scrollHeight - scrollerElement.scrollTop - scrollerElement.clientHeight > AFFORDANCE_THRESHOLD);
      }
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  // A new conversation or a new turn re-attaches; the user's own wheel or
  // touch scroll is the only thing that detaches.
  useEffect(() => {
    stuck.current = true;
    scrollToBottom();
  }, [conversation?.id, messages.length]);

  useEffect(() => {
    const element = scroller.current;
    if (!element) return;
    const detach = () => {
      const distance = element.scrollHeight - element.scrollTop - element.clientHeight;
      if (distance > STICK_THRESHOLD) stuck.current = false;
    };
    const onWheel = (event: WheelEvent) => {
      if (event.deltaY < 0) stuck.current = false;
      else window.setTimeout(detach, 0);
    };
    element.addEventListener("wheel", onWheel, { passive: true });
    element.addEventListener("touchmove", detach, { passive: true });
    return () => {
      element.removeEventListener("wheel", onWheel);
      element.removeEventListener("touchmove", detach);
    };
  }, []);

  const onScroll = () => {
    const element = scroller.current;
    if (!element) return;
    const distance = element.scrollHeight - element.scrollTop - element.clientHeight;
    // Reaching the bottom by any means re-attaches.
    if (distance <= STICK_THRESHOLD) stuck.current = true;
    setShowJump(distance > AFFORDANCE_THRESHOLD);
  };

  return (
    <div className="transcript" ref={scroller} onScroll={onScroll} aria-live="polite">
      {ready && messages.length === 0 && (mode === "agent" ? (
        <div className="empty">
          <div className="empty__badge">
            <Bot size={22} />
          </div>
          <h2>Give the agent a task</h2>
          <p>
            The model calls the enabled tools, their JavaScript handlers run in a sandbox, and the results go back to the model until it answers —
            without you pasting anything.
          </p>
          <div className="empty__grid">
            <button className="empty__card" onClick={() => setPanel("tools")}>
              Open the Tools panel to enable tools and write their handlers.
            </button>
          </div>
        </div>
      ) : (
        <div className="empty">
          <div className="empty__badge">
            <Sparkles size={22} />
          </div>
          <h2>What are we working on?</h2>
          <p>Ask anything. Streaming, thinking, tools, images, and three API protocols are all wired to the local runtime.</p>
          <div className="empty__grid">
            {SUGGESTIONS.map((text) => (
              <button key={text} className="empty__card" onClick={() => void sendMessage(text)}>
                {text}
              </button>
            ))}
          </div>
        </div>
      ))}
      <div className="transcript__inner" ref={inner}>
        {messages.map((message, index) => (
          <MessageItem key={message.id} message={message} previous={messages[index - 1]} isLast={index === messages.length - 1} />
        ))}
      </div>
      {showJump && (
        <button
          className="jump"
          onClick={() => {
            stuck.current = true;
            scrollToBottom("smooth");
          }}
          aria-label="Scroll to latest"
        >
          <ArrowDown size={16} />
        </button>
      )}
    </div>
  );
}
