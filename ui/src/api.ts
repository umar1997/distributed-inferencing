// Everything the page knows comes from the master's public API, like any other tool:
// - GET /api/events: live events (Server-Sent Events, moe_dist/master/events.py)
// - POST /v1/chat/completions: the OpenAI-style chat API, streamed
import { useEffect, useState } from "react";

export type NodeInfo = {
  node_id: string;
  role: "master" | "worker";
  address: string;
  status: "joining" | "healthy" | "slow" | "dead" | "left";
  reason: string;
  ready: boolean;
  experts: { all_layers: number[] };
  rtt_now_ms?: number | null;
  rtt_p95_ms?: number | null;
  ram_free_gb?: number | null;
  ram_total_gb?: number | null;
  memory_limit_gb?: number | null;
  cpu_cores?: number;
  cpu_arch?: string;
  calls_served?: number;
};

export type LayerRouting = {
  layer: number;
  experts: { id: number; node: string; weight?: number; count?: number }[];
  wait_ms: number;
  slowest: string | null;
};

export type TokenEvent = { req: string; token_index: number; text: string; special: boolean; layers: LayerRouting[] };

export type RequestEnd = {
  req: string;
  tokens: number;
  seconds: number;
  tokens_per_s: number;
  finish_reason: string | null;
  summary: string;
  node_time_ms: Record<string, { calls: number; compute_ms: number; net_ms: number }>;
};

export type Coverage = "ok" | "loading" | { missing: Record<string, number[]>; unavailable_nodes: Record<string, string> };

export type ClusterState = {
  connected: boolean;
  model: string;
  layers: number[];
  numExperts: number;
  placement: Record<string, (string | null)[]> | null; // layer -> node of each expert
  nodes: NodeInfo[];
  coverage: Coverage;
  request: { req: string; promptTokens: number; tokens: TokenEvent[]; end: RequestEnd | null } | null;
};

const EMPTY: ClusterState = {
  connected: false, model: "", layers: [], numExperts: 0, placement: null, nodes: [], coverage: "loading", request: null,
};

/** Listens to /api/events and keeps the latest picture of the cluster. EventSource reconnects by itself. */
export function useCluster(): ClusterState {
  const [state, setState] = useState<ClusterState>(EMPTY);
  useEffect(() => {
    const source = new EventSource("/api/events");
    source.onopen = () => setState((s) => ({ ...s, connected: true }));
    source.onerror = () => setState((s) => ({ ...s, connected: false }));
    source.onmessage = (message) => {
      const e = JSON.parse(message.data);
      setState((s) => {
        switch (e.type) {
          case "hello":
            return { ...s, connected: true, model: e.model, layers: e.layers, numExperts: e.num_experts,
                     placement: e.placement, nodes: e.nodes, coverage: e.coverage };
          case "node_update":
            return { ...s, nodes: e.nodes, coverage: e.coverage, placement: e.placement ?? s.placement };
          case "request_start":
            return { ...s, request: { req: e.req, promptTokens: e.prompt_tokens, tokens: [], end: null } };
          case "token":
            if (!s.request || s.request.req !== e.req) return s;
            return { ...s, request: { ...s.request, tokens: [...s.request.tokens, e] } };
          case "request_end":
            if (!s.request || s.request.req !== e.req) return s;
            return { ...s, request: { ...s.request, end: e } };
          default:
            return s;
        }
      });
    };
    return () => source.close();
  }, []);
  return state;
}

/** The node names in a stable order (master first), each with its own colour. */
export function nodeColors(nodes: NodeInfo[]): Record<string, string> {
  const palette = ["var(--node-0)", "var(--node-1)", "var(--node-2)", "var(--node-3)", "var(--node-4)", "var(--node-5)"];
  const colors: Record<string, string> = {};
  nodes.forEach((n, i) => (colors[n.node_id] = palette[i % palette.length]));
  return colors;
}

export type ChatMessage = { role: "user" | "assistant"; content: string; error?: string };

/** Streams one answer from /v1/chat/completions; calls onText for every new piece. */
export async function streamChat(
  messages: ChatMessage[],
  options: { temperature: number; maxTokens: number },
  onText: (text: string) => void,
  signal: AbortSignal,
): Promise<{ finishReason: string | null; error?: string }> {
  const reply = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: messages.map(({ role, content }) => ({ role, content })),
      stream: true, temperature: options.temperature, max_tokens: options.maxTokens,
    }),
    signal,
  });
  if (!reply.ok || !reply.body) {
    const body = await reply.json().catch(() => null);
    return { finishReason: null, error: body?.error?.message ?? `the master answered ${reply.status}` };
  }
  const reader = reply.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  let finishReason: string | null = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
      const data = JSON.parse(line.slice(6));
      if (data.error) return { finishReason, error: data.error.message };
      const choice = data.choices?.[0];
      if (choice?.delta?.content) onText(choice.delta.content);
      if (choice?.finish_reason) finishReason = choice.finish_reason;
    }
  }
  return { finishReason };
}

