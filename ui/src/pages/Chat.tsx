// The chat window: talks to the cluster through /v1/chat/completions, the same API any tool uses.
import { useEffect, useRef, useState } from "react";
import { ChatMessage, streamChat } from "../api";

export function Chat({ ready }: { ready: boolean }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [temperature, setTemperature] = useState(0);
  const [maxTokens, setMaxTokens] = useState(128);
  const stop = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  async function send() {
    const text = draft.trim();
    if (!text || busy) return;
    const history: ChatMessage[] = [...messages.filter((m) => !m.error), { role: "user", content: text }];
    setMessages([...history, { role: "assistant", content: "" }]);
    setDraft("");
    setBusy(true);
    stop.current = new AbortController();
    const addText = (piece: string) =>
      setMessages((all) => {
        const last = all[all.length - 1];
        return [...all.slice(0, -1), { ...last, content: last.content + piece }];
      });
    try {
      const result = await streamChat(history, { temperature, maxTokens }, addText, stop.current.signal);
      if (result.error) {
        setMessages((all) => [...all.slice(0, -1), { ...all[all.length - 1], error: result.error }]);
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setMessages((all) => [...all.slice(0, -1), { ...all[all.length - 1], error: String(err) }]);
      }
    } finally {
      setBusy(false);
      stop.current = null;
    }
  }

  return (
    <section className="panel chat">
      <div className="messages">
        {messages.length === 0 && (
          <p className="hint">
            Ask something. While the answer is written, the live view shows which experts each layer used and which
            node ran them.
          </p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            <div className="bubble">
              {m.content || (busy && i === messages.length - 1 ? <span className="typing">…</span> : null)}
            </div>
            {m.error && <div className="message-error">{m.error}</div>}
          </div>
        ))}
        <div ref={bottom} />
      </div>
      <form className="composer" onSubmit={(e) => (e.preventDefault(), send())}>
        <textarea
          value={draft}
          placeholder={ready ? "Message the cluster…" : "Waiting for the cluster to be ready…"}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          rows={2}
        />
        {busy ? (
          <button type="button" className="secondary" onClick={() => stop.current?.abort()}>Stop</button>
        ) : (
          <button type="submit" disabled={!draft.trim()}>Send</button>
        )}
      </form>
      <div className="options">
        <label>
          temperature <input type="number" min={0} max={2} step={0.1} value={temperature}
                             onChange={(e) => setTemperature(Number(e.target.value))} />
        </label>
        <label>
          max tokens <input type="number" min={1} max={2048} value={maxTokens}
                            onChange={(e) => setMaxTokens(Number(e.target.value))} />
        </label>
        <button type="button" className="link" onClick={() => setMessages([])} disabled={busy}>New chat</button>
      </div>
    </section>
  );
}
