// One card per node: health colour, address, memory, CPU, round trip time, experts held.
import { NodeInfo } from "../api";

function ranges(ids: number[]): string {
  const parts: string[] = [];
  for (let i = 0; i < ids.length; i++) {
    let j = i;
    while (j + 1 < ids.length && ids[j + 1] === ids[j] + 1) j++;
    parts.push(i === j ? `${ids[i]}` : `${ids[i]}–${ids[j]}`);
    i = j;
  }
  return parts.join(", ") || "none";
}

export function NodeMap({ nodes, colors }: { nodes: NodeInfo[]; colors: Record<string, string> }) {
  return (
    <div className="node-map">
      {nodes.map((n) => (
        <div key={n.node_id} className={`node-card status-${n.status}`} title={n.reason}>
          <div className="node-head">
            <span className="swatch" style={{ background: colors[n.node_id] }} />
            <strong>{n.node_id}</strong>
            <span className={`badge status-${n.status}`}>{n.status}</span>
          </div>
          <dl>
            <dt>address</dt>
            <dd>{n.role === "master" ? "this master" : n.address.replace(/^https?:\/\//, "")}</dd>
            <dt>experts</dt>
            <dd>{ranges(n.experts.all_layers)}</dd>
            {n.rtt_now_ms != null && (<><dt>round trip</dt><dd>{n.rtt_now_ms.toFixed(1)} ms</dd></>)}
            {n.ram_free_gb != null && (<><dt>RAM free</dt><dd>{n.ram_free_gb} GB</dd></>)}
            {n.cpu_cores != null && (<><dt>CPU</dt><dd>{n.cpu_cores} cores {n.cpu_arch ?? ""}</dd></>)}
          </dl>
          {n.status !== "healthy" && <div className="node-reason">{n.reason}</div>}
        </div>
      ))}
    </div>
  );
}
