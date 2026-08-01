"use client";

import { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";

type Source = {
  index: number;
  title: string;
  url: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

function SendIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
      <path
        d="M22 2L11 13M22 2L15 22l-4-9-9-4 20-7z"
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
        r="10"
        stroke="currentColor"
        strokeWidth="3"
        strokeOpacity="0.25"
      />
      <path
        d="M12 2a10 10 0 0 1 10 10"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}

function TypingDots() {
  return (
    <span className="typing-indicator">
      <span className="typing-dot" />
      <span className="typing-dot" />
      <span className="typing-dot" />
    </span>
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

    setMessages((prev) => [
      ...prev,
      { role: "user", content: userMessage },
    ]);

    setInput("");
    setLoading(true);

    // Placeholder assistant message
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", sources: [] },
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
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) return;

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
        const lines = event.split("\n");

        // Determine named event type, if any (e.g. "sources", "done")
        const eventTypeLine = lines.find((l) => l.startsWith("event: "));
        const eventType = eventTypeLine
          ? eventTypeLine.slice("event: ".length).trim()
          : null;

        // Collect ALL data: lines so multi-line payloads (like JSON arrays) work
        const dataPayload = lines
          .filter((l) => l.startsWith("data: "))
          .map((l) => l.slice("data: ".length))
          .join("\n");

        if (eventType === "sources") {
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
    <div className="chat-root">
      {/* ── Header ── */}
      <header className="chat-header">
        <div className="chat-header-left">
          <div className="chat-logo-box">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              <path
                d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <span className="chat-title">AI Assistant</span>
          <div className="chat-status">
            <span className="status-dot" />
            <span className="status-label">online</span>
          </div>
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

        {messages.map((msg, index) => (
          <div
            key={index}
            className={`msg-row msg-row--${msg.role}`}
          >
            <div className={`msg-bubble msg-bubble--${msg.role}`}>
              {/* Content or typing indicator */}
              {msg.content || <TypingDots />}

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
        ))}
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
