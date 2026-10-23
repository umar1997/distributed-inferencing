// Speed and where the time goes: per node, network time vs computing time (from request_end),
// and for the selected token, how long each layer waited for its experts.
import { RequestEnd, TokenEvent } from "../api";

export function Stats({ end, token, colors, live }: {
  end: RequestEnd | null; token: TokenEvent | null; colors: Record<string, string>; live: { tokens: number } | null;
}) {
  const layerWaits = token?.layers.map((l) => l.wait_ms) ?? [];
  const avgLayer = layerWaits.length ? layerWaits.reduce((a, b) => a + b, 0) / layerWaits.length : null;
  const nodes = Object.entries(end?.node_time_ms ?? {});
  const longest = Math.max(...nodes.map(([, t]) => t.compute_ms + t.net_ms), 1);

  return (
    <div className="stats">
      <div className="numbers">
        <Figure label="tokens / s" value={end ? end.tokens_per_s.toFixed(1) : live ? "…" : "–"} />
        <Figure label="tokens" value={end ? String(end.tokens) : live ? String(live.tokens) : "–"} />
        <Figure label="avg wait per layer" value={avgLayer !== null ? `${avgLayer.toFixed(1)} ms` : "–"}
                hint="for the selected token" />
        <Figure label="answer time" value={end ? `${end.seconds.toFixed(1)} s` : "–"} />
      </div>
      {nodes.length > 0 && (
        <div className="bars">
          <div className="bars-title">time per node for the last answer</div>
          {nodes.map(([node, t]) => (
            <div key={node} className="bar-row">
              <span className="bar-name"><span className="swatch" style={{ background: colors[node] }} />{node}</span>
              <span className="bar">
                <span className="bar-compute" style={{ width: `${(100 * t.compute_ms) / longest}%`, background: colors[node] }} />
                <span className="bar-net" style={{ width: `${(100 * t.net_ms) / longest}%` }} />
              </span>
              <span className="bar-value">
                {(t.compute_ms / 1000).toFixed(2)} s compute{t.net_ms > 0 ? ` + ${(t.net_ms / 1000).toFixed(2)} s network` : ""}
              </span>
            </div>
          ))}
          <div className="legend"><span className="bar-compute key" /> computing <span className="bar-net key" /> network
            (time waited minus the node's own computing time)</div>
        </div>
      )}
      {token && (
        <div className="layer-waits" title="how long each layer waited for all its experts, for the selected token">
          {token.layers.map((l) => (
            <span key={l.layer} className="layer-wait" style={{ height: `${Math.min(100, (100 * l.wait_ms) / Math.max(...layerWaits))}%` }}
                  title={`layer ${l.layer}: ${l.wait_ms.toFixed(1)} ms, slowest ${l.slowest ?? "–"}`} />
          ))}
        </div>
      )}
    </div>
  );
}

function Figure({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="number" title={hint}>
      <div className="value">{value}</div>
      <div className="label">{label}</div>
    </div>
  );
}
