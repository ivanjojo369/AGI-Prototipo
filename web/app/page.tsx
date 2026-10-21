// web/app/page.tsx
"use client";

import { useEffect, useRef, useState } from "react";

type Msg = {
  role: "user" | "assistant" | "system";
  content: string;
};

type ConversationSummary = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
};

type Conversation = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: Msg[];
};

const LAST_CHAT_KEY = "quipu:last-conversation-id";

function pickAssistantText(data: any): string {
  if (typeof data === "string") return data;

  return (
    data?.reply ??
    data?.text ??
    data?.message ??
    data?.content ??
    data?.assistant ??
    data?.output ??
    data?.choices?.[0]?.message?.content ??
    data?.choices?.[0]?.text ??
    data?.result ??
    data?.data?.reply ??
    data?.data?.text ??
    JSON.stringify(data)
  );
}

function shortTitle(title: string): string {
  const clean = String(title || "Nuevo chat").trim();
  return clean.length > 34
    ? `${clean.slice(0, 31)}...`
    : clean;
}

export default function Page() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [conversations, setConversations] =
    useState<ConversationSummary[]>([]);
  const [activeConversationId, setActiveConversationId] =
    useState("");
  const [activeTitle, setActiveTitle] = useState("Chat");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingChats, setLoadingChats] = useState(true);
  const [showBlockMenu, setShowBlockMenu] = useState(false);

  const endRef = useRef<HTMLDivElement>(null);
  const blockMenuRef = useRef<HTMLDivElement>(null);

  const userBlocks = messages.filter(
    (m) => m.role === "user"
  ).length;
  const assistantBlocks = messages.filter(
    (m) => m.role === "assistant"
  ).length;
  const totalBlocks = userBlocks + assistantBlocks;

  useEffect(() => {
    void initializeChats();
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        blockMenuRef.current &&
        !blockMenuRef.current.contains(event.target as Node)
      ) {
        setShowBlockMenu(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () =>
      document.removeEventListener(
        "mousedown",
        handleClickOutside
      );
  }, []);

  async function fetchConversationList():
    Promise<ConversationSummary[]> {
    const res = await fetch("/api/chat?action=list", {
      method: "GET",
      cache: "no-store",
    });

    if (!res.ok) {
      throw new Error(
        "No se pudo cargar la lista de conversaciones."
      );
    }

    const data = await res.json();

    return Array.isArray(data?.conversations)
      ? data.conversations
      : [];
  }

  async function initializeChats() {
    setLoadingChats(true);

    try {
      const list = await fetchConversationList();
      setConversations(list);

      if (list.length === 0) {
        await createNewChat();
        return;
      }

      const stored =
        typeof window !== "undefined"
          ? window.localStorage.getItem(LAST_CHAT_KEY)
          : null;

      const selected =
        list.find((item) => item.id === stored) ??
        list[0];

      await loadConversation(selected.id, list);
    } catch {
      setMessages([
        {
          role: "assistant",
          content:
            "No pude cargar tus conversaciones. ¿Está activo quipu_http.py?",
        },
      ]);
    } finally {
      setLoadingChats(false);
    }
  }

  async function refreshConversationList() {
    try {
      const list = await fetchConversationList();
      setConversations(list);

      const current = list.find(
        (item) => item.id === activeConversationId
      );

      if (current) {
        setActiveTitle(current.title);
      }
    } catch {
      // El chat activo puede seguir funcionando.
    }
  }

  async function loadConversation(
    id: string,
    knownList?: ConversationSummary[]
  ) {
    if (!id) return;

    setLoadingChats(true);

    try {
      const res = await fetch(
        `/api/chat?action=load&id=${encodeURIComponent(id)}`,
        {
          method: "GET",
          cache: "no-store",
        }
      );

      if (!res.ok) {
        throw new Error("No se pudo abrir la conversación.");
      }

      const data = await res.json();
      const conversation: Conversation | null =
        data?.conversation ?? null;

      if (!conversation) {
        throw new Error("Conversación inválida.");
      }

      setActiveConversationId(conversation.id);
      setActiveTitle(conversation.title || "Chat");
      setMessages(
        Array.isArray(conversation.messages)
          ? conversation.messages
          : []
      );
      setInput("");
      setShowBlockMenu(false);

      if (typeof window !== "undefined") {
        window.localStorage.setItem(
          LAST_CHAT_KEY,
          conversation.id
        );
      }

      if (knownList) {
        setConversations(knownList);
      }
    } catch {
      setMessages([
        {
          role: "assistant",
          content: "No pude abrir esta conversación.",
        },
      ]);
    } finally {
      setLoadingChats(false);
    }
  }

  async function createNewChat() {
    setLoadingChats(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          action: "create_conversation",
          title: "Nuevo chat",
        }),
      });

      if (!res.ok) {
        throw new Error(
          "No se pudo crear la conversación."
        );
      }

      const data = await res.json();
      const conversation: Conversation | null =
        data?.conversation ?? null;

      if (!conversation) {
        throw new Error("Conversación inválida.");
      }

      setActiveConversationId(conversation.id);
      setActiveTitle(
        conversation.title || "Nuevo chat"
      );
      setMessages(
        Array.isArray(conversation.messages)
          ? conversation.messages
          : []
      );
      setInput("");

      if (typeof window !== "undefined") {
        window.localStorage.setItem(
          LAST_CHAT_KEY,
          conversation.id
        );
      }

      const list = await fetchConversationList();
      setConversations(list);
    } catch {
      setMessages([
        {
          role: "assistant",
          content: "No pude crear un nuevo chat.",
        },
      ]);
    } finally {
      setLoadingChats(false);
    }
  }

  async function send() {
    const text = input.trim();

    if (
      !text ||
      loading ||
      loadingChats ||
      !activeConversationId
    ) {
      return;
    }

    setInput("");
    setMessages((current) => [
      ...current,
      { role: "user", content: text },
    ]);
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          action: "send_message",
          conversation_id: activeConversationId,
          message: text,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));

        setMessages((current) => [
          ...current,
          {
            role: "assistant",
            content:
              err?.detail ??
              "No pude obtener una respuesta de Quipu.",
          },
        ]);

        return;
      }

      const data = await res.json().catch(
        async () => ({
          reply: await res.text(),
        })
      );

      const reply = pickAssistantText(data);

      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: reply,
        },
      ]);

      if (data?.conversation?.title) {
        setActiveTitle(
          data.conversation.title
        );
      }

      await refreshConversationList();
    } catch {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content:
            "Error de red. ¿Está activo tu backend?",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "260px 1fr",
        minHeight: "100vh",
      }}
    >
      <aside
        style={{
          background: "#fff",
          borderRight: "1px solid #eaeaea",
          display: "flex",
          flexDirection: "column",
          minHeight: "100vh",
        }}
      >
        <div
          style={{
            height: 56,
            display: "flex",
            alignItems: "center",
            padding: "0 12px",
            borderBottom: "1px solid #eee",
            fontWeight: 600,
          }}
        >
          Synchronexis
        </div>

        <div style={{ padding: 12 }}>
          <button
            type="button"
            onClick={() => void createNewChat()}
            disabled={loadingChats}
            style={{
              width: "100%",
              border: "1px solid #d9dce3",
              borderRadius: 10,
              background: "#fff",
              padding: "9px 10px",
              textAlign: "left",
              fontWeight: 600,
              cursor: loadingChats ? "wait" : "pointer",
              marginBottom: 14,
            }}
          >
            + Nuevo chat
          </button>

          <div
            style={{
              marginBottom: 8,
              fontSize: 13,
              color: "#777",
              fontWeight: 600,
            }}
          >
            Conversaciones
          </div>

          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 6,
              maxHeight: "calc(100vh - 150px)",
              overflowY: "auto",
            }}
          >
            {conversations.map((conversation) => {
              const active =
                conversation.id === activeConversationId;

              return (
                <button
                  key={conversation.id}
                  type="button"
                  onClick={() =>
                    void loadConversation(
                      conversation.id
                    )
                  }
                  title={conversation.title}
                  style={{
                    width: "100%",
                    border: active
                      ? "1px solid #d9dce3"
                      : "1px solid transparent",
                    borderRadius: 9,
                    background: active
                      ? "#f3f4f6"
                      : "transparent",
                    padding: "8px 9px",
                    textAlign: "left",
                    cursor: "pointer",
                    color: "#222",
                  }}
                >
                  <div
                    style={{
                      fontSize: 13,
                      fontWeight: active ? 650 : 500,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {shortTitle(
                      conversation.title
                    )}
                  </div>

                  <div
                    style={{
                      marginTop: 2,
                      fontSize: 11,
                      color: "#999",
                    }}
                  >
                    {conversation.message_count} bloques
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </aside>

      <main
        style={{
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
        }}
      >
        <div
          style={{
            height: 56,
            background: "#fff",
            borderBottom: "1px solid #eaeaea",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "0 16px",
            position: "relative",
            zIndex: 20,
          }}
        >
          <div
            style={{
              fontWeight: 600,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              maxWidth: "60%",
            }}
            title={activeTitle}
          >
            {activeTitle || "Chat"}
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
            }}
          >
            <div
              style={{
                fontSize: 12,
                color: "#666",
              }}
            >
              AGI doméstica textual
            </div>

            <div
              ref={blockMenuRef}
              style={{ position: "relative" }}
            >
              <button
                type="button"
                aria-label="Abrir información del chat"
                aria-expanded={showBlockMenu}
                onClick={() =>
                  setShowBlockMenu(
                    (open) => !open
                  )
                }
                style={{
                  width: 36,
                  height: 36,
                  border: "1px solid #e5e7eb",
                  borderRadius: 10,
                  background: "#fff",
                  color: "#111",
                  cursor: "pointer",
                  fontSize: 22,
                  lineHeight: 1,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  padding: 0,
                }}
              >
                ⋮
              </button>

              {showBlockMenu && (
                <div
                  style={{
                    position: "absolute",
                    top: 44,
                    right: 0,
                    width: 220,
                    background: "#fff",
                    border: "1px solid #e5e7eb",
                    borderRadius: 14,
                    boxShadow:
                      "0 12px 30px rgba(0,0,0,0.12)",
                    padding: 14,
                    zIndex: 100,
                  }}
                >
                  <div
                    style={{
                      fontWeight: 700,
                      fontSize: 14,
                      marginBottom: 12,
                    }}
                  >
                    Bloques del chat
                  </div>

                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns:
                        "1fr auto",
                      gap: "8px 14px",
                      fontSize: 13,
                    }}
                  >
                    <span
                      style={{ color: "#666" }}
                    >
                      Total
                    </span>
                    <strong>{totalBlocks}</strong>

                    <span
                      style={{ color: "#666" }}
                    >
                      Tú
                    </span>
                    <strong>{userBlocks}</strong>

                    <span
                      style={{ color: "#666" }}
                    >
                      Quipu
                    </span>
                    <strong>{assistantBlocks}</strong>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div
          style={{
            flex: 1,
            overflowY: "auto",
            padding: 16,
          }}
        >
          <div
            style={{
              maxWidth: 860,
              margin: "0 auto",
            }}
          >
            {messages.map(
              (message, index) => (
                <div
                  key={index}
                  style={{
                    display: "flex",
                    justifyContent:
                      message.role === "user"
                        ? "flex-end"
                        : "flex-start",
                    marginBottom: 8,
                  }}
                >
                  <div
                    style={{
                      background:
                        message.role === "user"
                          ? "#4f46e5"
                          : "#fff",
                      color:
                        message.role === "user"
                          ? "#fff"
                          : "#111",
                      border:
                        message.role === "user"
                          ? "none"
                          : "1px solid #eaeaea",
                      padding: "8px 12px",
                      borderRadius: 16,
                      maxWidth: "90%",
                      whiteSpace: "pre-wrap",
                      lineHeight: 1.45,
                    }}
                  >
                    {message.content}
                  </div>
                </div>
              )
            )}

            <div ref={endRef} />
          </div>
        </div>

        <div
          style={{
            borderTop: "1px solid #eaeaea",
            background: "#fff",
            padding: 12,
          }}
        >
          <div
            style={{
              maxWidth: 860,
              margin: "0 auto",
              display: "flex",
              gap: 8,
            }}
          >
            <textarea
              value={input}
              onChange={(event) =>
                setInput(event.target.value)
              }
              onKeyDown={(event) => {
                if (
                  event.key === "Enter" &&
                  !event.shiftKey
                ) {
                  event.preventDefault();
                  void send();
                }
              }}
              placeholder={
                activeConversationId
                  ? "Escribe tu mensaje…"
                  : "Cargando chat…"
              }
              disabled={
                loading ||
                loadingChats ||
                !activeConversationId
              }
              rows={1}
              style={{
                flex: 1,
                resize: "none",
                border: "1px solid #ddd",
                borderRadius: 12,
                padding: "10px 12px",
                outline: "none",
              }}
            />

            <button
              onClick={() => void send()}
              disabled={
                loading ||
                loadingChats ||
                !activeConversationId
              }
              style={{
                background:
                  loading || loadingChats
                    ? "#9aa2ff"
                    : "#4f46e5",
                color: "#fff",
                border: "none",
                borderRadius: 12,
                padding: "10px 14px",
                cursor:
                  loading || loadingChats
                    ? "not-allowed"
                    : "pointer",
                minWidth: 88,
                fontWeight: 600,
              }}
            >
              {loading
                ? "Enviando…"
                : "Enviar"}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
