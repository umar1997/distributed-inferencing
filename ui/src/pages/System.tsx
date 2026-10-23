// The live view of the cluster: nodes, the expert grid, and the selected token's routing.
import { useEffect, useState } from "react";
import { ClusterState, nodeColors } from "../api";
import { ExpertGrid } from "../components/ExpertGrid";
import { NodeMap } from "../components/NodeMap";
import { Stats } from "../components/Stats";

export function System({ cluster }: { cluster: ClusterState }) {
  const colors = nodeColors(cluster.nodes);
  const tokens = cluster.request?.tokens ?? [];
  const generating = cluster.request !== null && cluster.request.end === null;
  // While "live", the newest token is shown; moving the slider or clicking a token pins one.
  const [live, setLive] = useState(true);
  const [picked, setPicked] = useState(0);
  useEffect(() => {
    setLive(true);
  }, [cluster.request?.req]);
  const index = live ? tokens.length - 1 : Math.min(picked, tokens.length - 1);
  const token = index >= 0 ? tokens[index] : null;

  const pick = (i: number) => {
    setPicked(i);
    setLive(i === tokens.length - 1 && generating);
  };

  return (
    <section className="panel system">
      <Coverage coverage={cluster.coverage} connected={cluster.connected} />
      <NodeMap nodes={cluster.nodes} colors={colors} />

      <div className="routing">
        <div className="routing-head">
          <h2>Experts per layer</h2>
          {tokens.length > 0 && (
            <div className="slider">
              <input type="range" min={0} max={tokens.length - 1} value={Math.max(index, 0)}
                     onChange={(e) => pick(Number(e.target.value))} aria-label="token to show" />
              <span className="slider-label">
                {index === 0 ? "prompt pass → token 1" : `token ${index + 1}`} of {tokens.length}
                <span title="copy this into a log search: docker compose logs | grep '…'">
                  {" · logs: "}<span className="req">req={cluster.request?.req} token={index}</span>
                </span>
                {generating && (live ? " · live" : <button className="link" onClick={() => setLive(true)}>follow live</button>)}
              </span>
            </div>
          )}
        </div>
        {tokens.length > 0 && (
          <div className="token-strip" aria-label="the answer, token by token">
            {tokens.map((t, i) => (
              <button key={i} className={`token${i === index ? " current" : ""}${t.special ? " special" : ""}`}
                      onClick={() => pick(i)} title={`token ${i + 1}`}>
                {t.special ? "⏹" : t.text.replace(/\n/g, "↵") || "·"}
              </button>
            ))}
          </div>
        )}
        <ExpertGrid layers={cluster.layers} numExperts={cluster.numExperts} placement={cluster.placement}
                    colors={colors} token={token} />
      </div>

      <Stats end={cluster.request?.end ?? null} token={token} colors={colors}
             live={generating ? { tokens: tokens.length } : null} />
    </section>
  );
}

function Coverage({ coverage, connected }: { coverage: ClusterState["coverage"]; connected: boolean }) {
  if (!connected) return <div className="banner warn">Not connected to the master: trying again…</div>;
  if (coverage === "ok") return null;
  if (coverage === "loading") return <div className="banner warn">The master is still loading.</div>;
  const missing = Object.entries(coverage.missing).map(([where, ids]) =>
    `${where === "all_layers" ? "in every layer" : `layer ${where}`}: ${ids.join(", ")}`).join("; ");
  const holders = Object.entries(coverage.unavailable_nodes).map(([n, s]) => `${n} (${s})`).join(", ");
  return (
    <div className="banner error">
      Missing experts {missing}. {holders ? `Held by ${holders}.` : "No node holds them."} Chat requests are refused
      until they are back.
    </div>
  );
}
