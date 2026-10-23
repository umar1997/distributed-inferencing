// Layers (rows) × experts (columns). Each cell has the colour of the node that holds that expert;
// the experts picked for the selected token light up (brighter = the router counted it more).
import { TokenEvent } from "../api";

type Props = {
  layers: number[];
  numExperts: number;
  placement: Record<string, (string | null)[]> | null;
  colors: Record<string, string>;
  token: TokenEvent | null;
};

export function ExpertGrid({ layers, numExperts, placement, colors, token }: Props) {
  // layer -> expert -> how strongly it was used (0..1), for the selected token
  const used = new Map<number, Map<number, number>>();
  let isPrompt = false;
  for (const layer of token?.layers ?? []) {
    const row = new Map<number, number>();
    const most = Math.max(...layer.experts.map((e) => e.count ?? 0), 1);
    for (const e of layer.experts) {
      if (e.count !== undefined) isPrompt = true;
      // weights of 8 experts add up to 1, so 0.125 is "average": scale so that reads as mid-bright
      row.set(e.id, e.weight !== undefined ? Math.min(1, 0.45 + e.weight * 2.2) : 0.3 + 0.7 * (e.count! / most));
    }
    used.set(layer.layer, row);
  }
  const experts = Array.from({ length: numExperts }, (_, i) => i);

  return (
    <div className="grid-wrap">
      <div className="expert-grid" style={{ gridTemplateColumns: `2.2em repeat(${numExperts}, minmax(0, 1fr))` }}>
        <div className="corner">L \ E</div>
        {experts.map((e) => (
          <div key={e} className="col-label">{e % 4 === 0 ? e : ""}</div>
        ))}
        {layers.map((layer) => (
          <Row key={layer} layer={layer} experts={experts} owners={placement?.[String(layer)] ?? []}
               colors={colors} used={used.get(layer)} lit={token !== null} />
        ))}
      </div>
      <p className="grid-caption">
        {token === null
          ? "Each cell is one expert in one layer, coloured by the node that holds it."
          : isPrompt
            ? "The pass over the prompt: every expert any prompt token used (brighter = used by more tokens)."
            : "Lit cells: the 8 experts the router picked in each layer for this token (brighter = counted more)."}
      </p>
    </div>
  );
}

function Row({ layer, experts, owners, colors, used, lit }: {
  layer: number; experts: number[]; owners: (string | null)[]; colors: Record<string, string>;
  used: Map<number, number> | undefined; lit: boolean;
}) {
  return (
    <>
      <div className="row-label">{layer}</div>
      {experts.map((e) => {
        const owner = owners[e];
        const strength = used?.get(e);
        const style = {
          background: owner ? colors[owner] ?? "var(--muted)" : "var(--missing)",
          opacity: !lit ? 0.55 : strength !== undefined ? strength : 0.12,
        };
        return (
          <div key={e} className={`cell${strength !== undefined ? " picked" : ""}`} style={style}
               title={`layer ${layer}, expert ${e}: ${owner ?? "nobody"}${strength !== undefined ? " (picked)" : ""}`} />
        );
      })}
    </>
  );
}
