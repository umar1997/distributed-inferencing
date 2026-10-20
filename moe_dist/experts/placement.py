"""Which node holds which expert, in which layer, and whether every expert is covered.

The map is always "layer -> expert -> node(s)". `EXPERT_IDS=0-10` is a shortcut that puts the
same expert numbers in every layer (Phase 4.3 adds per-layer maps). Expert 5 in layer 0 has
nothing to do with expert 5 in layer 1; the numbers are just positions.
"""

import torch

from moe_dist.experts.slicing import check_expert_ids


def parse_expert_ids(text: str, num_experts: int) -> list[int]:
    """'0-10' / '0,4,8,12' / '0-3,8,12-15' / 'none' -> sorted list of expert numbers."""
    text = text.strip().lower()
    if text in ("", "none"):
        return []
    ids = []
    for piece in text.split(","):
        piece = piece.strip()
        try:
            if "-" in piece:
                first, last = (int(x) for x in piece.split("-"))
                if first > last:
                    raise ValueError
                ids.extend(range(first, last + 1))
            else:
                ids.append(int(piece))
        except ValueError:
            raise ValueError(f"cannot read expert list {text!r}: bad part {piece!r} (use e.g. '0-10' or '0,4,8')") from None
    return check_expert_ids(ids, num_experts)


def format_expert_ids(ids: list[int]) -> str:
    """The reverse of parse_expert_ids: [10, 11, 12, 20] -> '10-12,20' ('none' when empty)."""
    parts, start = [], None
    for i, e in enumerate(ids):
        if start is None:
            start = e
        if i + 1 == len(ids) or ids[i + 1] != e + 1:
            parts.append(f"{start}-{e}" if e != start else f"{e}")
            start = None
    return ",".join(parts) or "none"


class CoverageError(ValueError):
    """Some experts are held by no node, so the model cannot run."""


class Placement:
    def __init__(self, layers: list[int], num_experts: int, owners: dict[int, dict[int, list[str]]]):
        """owners[layer][expert] = node IDs holding that expert (first one is used)."""
        self.layers = list(layers)
        self.num_experts = num_experts
        self.owners = owners

    @classmethod
    def same_in_every_layer(cls, node_experts: dict[str, list[int]], layers: list[int], num_experts: int):
        """{node_id: expert numbers} -> the same experts on that node in every layer."""
        owners = {layer: {e: [] for e in range(num_experts)} for layer in layers}
        for node_id, ids in node_experts.items():
            for e in check_expert_ids(ids, num_experts) if ids else []:
                for layer in layers:
                    owners[layer][e].append(node_id)
        return cls(layers, num_experts, owners)

    def node_ids(self) -> list[str]:
        """Every node that holds at least one expert, in first-seen order."""
        seen = {}
        for layer in self.layers:
            for nodes in self.owners[layer].values():
                seen.update(dict.fromkeys(nodes))
        return list(seen)

    def experts_of(self, node_id: str, layer: int) -> list[int]:
        return [e for e, nodes in self.owners[layer].items() if node_id in nodes]

    def missing(self) -> dict[int, list[int]]:
        """{layer: experts no node holds}, only layers with gaps."""
        gaps = {layer: [e for e, nodes in self.owners[layer].items() if not nodes] for layer in self.layers}
        return {layer: ids for layer, ids in gaps.items() if ids}

    def check_coverage(self):
        gaps = self.missing()
        if not gaps:
            return
        if len(gaps) == len(self.layers) and len({tuple(ids) for ids in gaps.values()}) == 1:
            raise CoverageError(f"missing experts: {next(iter(gaps.values()))} in all layers")
        detail = "; ".join(f"layer {layer}: {ids}" for layer, ids in gaps.items())
        raise CoverageError(f"missing experts: {detail}")

    def owner_table(self, layer: int) -> tuple[torch.Tensor, list[str]]:
        """For fast lookups: (table, node_ids) where table[e] = index into node_ids, or -1."""
        node_ids = self.node_ids()
        table = torch.full((self.num_experts,), -1, dtype=torch.long)
        for e, nodes in self.owners[layer].items():
            if nodes:
                table[e] = node_ids.index(nodes[0])
        return table, node_ids
