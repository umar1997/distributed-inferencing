// Wide screens: chat on the left, the live system view on the right (watch the experts while the
// answer is written). Narrow screens (phones): two tabs.
import { useState } from "react";
import { useCluster } from "./api";
import { Chat } from "./pages/Chat";
import { System } from "./pages/System";

export function App() {
  const cluster = useCluster();
  const [tab, setTab] = useState<"chat" | "system">("chat");
  const healthy = cluster.nodes.filter((n) => n.status === "healthy" || n.status === "slow").length;

  return (
    <div className="app">
      <header>
        <h1>MoE cluster</h1>
        <span className="model">{cluster.model.split("/").pop()}</span>
        <span className={`pill ${cluster.connected && cluster.coverage === "ok" ? "ok" : "warn"}`}>
          {!cluster.connected ? "disconnected" : cluster.coverage === "ok" ? "ready" : "not ready"} · {healthy}/
          {cluster.nodes.length} nodes
        </span>
        <nav className="tabs">
          <button className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}>Chat</button>
          <button className={tab === "system" ? "active" : ""} onClick={() => setTab("system")}>System</button>
        </nav>
      </header>
      <main className={`layout show-${tab}`}>
        <Chat ready={cluster.coverage === "ok"} />
        <System cluster={cluster} />
      </main>
    </div>
  );
}
