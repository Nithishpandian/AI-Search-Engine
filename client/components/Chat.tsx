"use client";

import { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";

type Source = {
  index: number;
  title: string;
  url: string;
};

type ResearchStep = {
  step: number;
  query: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  researchSteps?: ResearchStep[];
};

function SendIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
      <path
        d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function SpinnerIcon() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      style={{ animation: "spin 0.65s linear infinite" }}
    >
      <circle
        cx="12"
        cy="12"
        r="9"
        stroke="currentColor"
        strokeWidth="2"
        strokeOpacity="0.25"
      />
      <path
        d="M21 12a9 9 0 0 0-9-9"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

function TypingDots() {
  return (
    <span className="typing-dots">
      <span></span>
      <span></span>
      <span></span>
    </span>
  );
}

function ResearchStatus({ steps }: { steps: ResearchStep[] }) {
  const latest = steps[steps.length - 1];
  return (
    <div className="research-status">
      <SpinnerIcon />
      <span className="research-status-text">
        Researching — step {latest.step}: “{latest.query}”
      </span>
    </div>
  );
}

export default function Chat() {
  const { token, logout } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function sendMessage() {
    if (!input.trim() || loading) return;

    const userMessage = input;

    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);

    setInput("");
    setLoading(true);

    // Placeholder assistant message
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", sources: [], researchSteps: [] },
    ]);

    const response = await fetch("http://localhost:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        message: userMessage,
        conversation_id: conversationId,
      }),
    });

    if (response.status === 401) {
      logout();
      setLoading(false);
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      setLoading(false);
      return;
    }

    const decoder = new TextDecoder();
    let assistantText = "";
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Split on double-newline SSE event boundaries
      const events = buffer.split("\n\n");
      // Last element may be an incomplete event — keep it for the next chunk
      buffer = events.pop() ?? "";

      for (const event of events) {
        if (!event.trim()) continue;

        const lines = event.split("\n");

        // Determine named event type, if any
        // (e.g. "research_step", "sources", "done")
        const eventTypeLine = lines.find((l) => l.startsWith("event: "));
        const eventType = eventTypeLine
          ? eventTypeLine.slice("event: ".length).trim()
          : null;

        // Collect ALL data: lines so multi-line payloads (like JSON arrays) work
        const dataPayload = lines
          .filter((l) => l.startsWith("data: "))
          .map((l) => l.slice("data: ".length))
          .join("\n");

        if (eventType === "research_step") {
          try {
            const step: ResearchStep = JSON.parse(dataPayload);
            setMessages((prev) => {
              const copy = [...prev];
              const last = copy[copy.length - 1];
              copy[copy.length - 1] = {
                ...last,
                researchSteps: [...(last.researchSteps ?? []), step],
              };
              return copy;
            });
          } catch {
            // ignore malformed payload
          }
        } else if (eventType === "sources") {
          try {
            const sources: Source[] = JSON.parse(dataPayload);
            setMessages((prev) => {
              const copy = [...prev];
              copy[copy.length - 1] = {
                ...copy[copy.length - 1],
                sources,
              };
              return copy;
            });
          } catch {
            // ignore malformed payload
          }
        } else if (eventType === "done") {
          const id = dataPayload.trim();
          if (id) setConversationId(id);
        } else if (eventType === null && dataPayload) {
          // Plain data event — streamed assistant text
          assistantText += dataPayload;
          setMessages((prev) => {
            const copy = [...prev];
            copy[copy.length - 1] = {
              ...copy[copy.length - 1],
              content: assistantText,
            };
            return copy;
          });
        }
      }
    }

    setLoading(false);
  }

  return (
    <div className="chat-container">
      {/* ── Header ── */}
      <header className="chat-header">
        <div className="chat-header-info">
          <span className="chat-header-title">AI Assistant</span>
          <span className="chat-header-status">online</span>
        </div>

        <button
          id="logout-btn"
          onClick={logout}
          className="logout-btn"
          title="Sign out"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
            <path
              d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          Sign out
        </button>
      </header>

      {/* ── Messages ── */}
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                <path
                  d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
            <p className="empty-title">Start a conversation</p>
            <p className="empty-sub">// ask anything</p>
          </div>
        )}

        {messages.map((msg, index) => {
          const isStreamingAssistant =
            msg.role === "assistant" &&
            !msg.content &&
            index === messages.length - 1;

          const showResearchStatus =
            isStreamingAssistant &&
            msg.researchSteps &&
            msg.researchSteps.length > 0;

          return (
            <div key={index} className={`msg-row msg-row--${msg.role}`}>
              <div className={`msg-bubble msg-bubble--${msg.role}`}>
                {/* Content, research status, or typing indicator */}
                {msg.content ? (
                  msg.content
                ) : showResearchStatus ? (
                  <ResearchStatus steps={msg.researchSteps!} />
                ) : (
                  <TypingDots />
                )}

                {/* Sources — only for assistant */}
                {msg.role === "assistant" &&
                  msg.sources &&
                  msg.sources.length > 0 && (
                    <div className="sources-section">
                      <p className="sources-heading">Sources</p>
                      <div className="sources-list">
                        {msg.sources.map((source) => (
                          <a
                            key={source.index}
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="source-link"
                          >
                            <span className="source-num">[{source.index}]</span>
                            {source.title}
                          </a>
                        ))}
                      </div>
                    </div>
                  )}
              </div>
            </div>
          );
        })}
      </div>

      {/* ── Input bar ── */}
      <div className="chat-input-bar">
        <input
          id="chat-input"
          className="chat-input"
          placeholder="Ask anything..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              sendMessage();
            }
          }}
        />
        <button
          id="chat-send-btn"
          onClick={sendMessage}
          disabled={loading}
          className="send-btn"
          title="Send"
        >
          {loading ? <SpinnerIcon /> : <SendIcon />}
        </button>
      </div>
    </div>
  );
}