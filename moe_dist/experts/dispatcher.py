"""The dispatcher: replaces a layer's `experts` module and sends the work to the nodes.

Transformers calls `self.experts(hidden_states, top_k_index, top_k_weights)` in every MoE layer.
The dispatcher takes the same inputs and gives the same output, but:

1. turns the router's picks into a work list of (token, expert, weight) pairs (`plan_pairs`),
2. gives each node only its pairs and only the token rows those pairs need (one message per node
   per layer, packed with transport/codec.py),
3. puts the answers back together (`combine_exact` or the fast per-token sums).

A node is anything with `node_id` and `forward(message: bytes) -> bytes`: a `LocalNode` in the same
program (the master's own experts, or Phase 1's virtual nodes), or a `RemoteNode`
(transport/client.py) that sends the bytes to a worker over HTTP. The bytes are the same either way.

With an `executor` (a thread pool), all remote nodes of a layer are called at the same time while
the master runs its own experts; the answers are still combined in the fixed order, so running
in parallel cannot change the result.
"""

import contextvars
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from moe_dist.experts.compute import EXACT, FAST, MODES, ExactResult, Pairs, combine_exact, plan_pairs, run_pairs
from moe_dist.transport import codec
from moe_dist.transport.client import NodeError


@dataclass
class RequestInfo:
    """Per-request facts the dispatcher puts into every message (and later into the logs)."""

    request_id: str
    token_index: int = -1  # forward pass number: 0 = the prompt, 1 = first new token, ...


# A "context variable" is a global that each request (thread / async task) sees separately,
# so Transformers' code does not need to pass these facts through.
current_request: contextvars.ContextVar[RequestInfo | None] = contextvars.ContextVar("current_request", default=None)


@contextmanager
def request_scope(request_id: str):
    info = RequestInfo(request_id)
    token = current_request.set(info)
    try:
        yield info
    finally:
        current_request.reset(token)


def serve_forward(node_id: str, parts: dict, act_fn, message: bytes) -> bytes:
    """What a node does with one request message: unpack, run its experts, pack the answer.

    `parts` is {layer: ExpertSlice}. Shared by LocalNode now and the worker service in Phase 2.
    """
    header, t = codec.decode(message)
    layer, mode = header["layer"], header["combine_mode"]
    if layer not in parts:
        raise ValueError(f"node {node_id} holds no experts for layer {layer}")
    hidden = t["hidden"]
    pairs = Pairs(row=t["row"].long(), expert=t["expert"].long(), weight=t["weight"])
    start = time.perf_counter()
    result = run_pairs(parts[layer], act_fn, hidden, pairs, mode)
    compute_ms = (time.perf_counter() - start) * 1000
    out = result.rows if mode == EXACT else result.sums
    reply = {
        "node_id": node_id,
        "request_id": header.get("request_id"),
        "layer": layer,
        "compute_ms": round(compute_ms, 3),
        "experts_run": sorted(set(pairs.expert.tolist())),
    }
    return codec.encode(reply, {"out": out})


class LocalNode:
    """A "virtual node": holds some experts and answers messages, inside this program."""

    def __init__(self, node_id: str, parts: dict, act_fn):
        self.node_id = node_id
        self.parts = parts  # {layer: ExpertSlice}
        self.act_fn = act_fn
        self.calls = 0

    def forward(self, message: bytes) -> bytes:
        self.calls += 1
        return serve_forward(self.node_id, self.parts, self.act_fn, message)

    def nbytes(self) -> int:
        return sum(p.nbytes() for p in self.parts.values())


class ExpertDispatcher(nn.Module):
    """Drop-in replacement for one layer's `experts` module."""

    def __init__(
        self, layer: int, placement, nodes: dict, combine_mode: str = EXACT, first_layer: bool = False, executor=None
    ):
        super().__init__()
        if combine_mode not in MODES:
            raise ValueError(f"unknown combine mode {combine_mode!r}; use one of {MODES}")
        self.layer = layer
        self.combine_mode = combine_mode
        self.first_layer = first_layer  # counts forward passes for RequestInfo.token_index
        table, node_ids = placement.owner_table(layer)
        missing = [n for n in node_ids if n not in nodes]
        if missing:
            raise ValueError(f"layer {layer}: placement names nodes that do not exist: {missing}")
        self.owner_of = table  # expert -> index into self.node_ids, -1 = nobody
        self.node_ids = node_ids
        self.nodes = nodes
        self.executor = executor
        self.stats = Counter()  # calls, bytes_sent, bytes_received

    def extra_repr(self) -> str:
        return f"layer={self.layer}, nodes={self.node_ids}, combine_mode={self.combine_mode}"

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        info = current_request.get()
        if info is not None and self.first_layer:
            info.token_index += 1
        pairs = plan_pairs(top_k_index, top_k_weights)
        owner = self.owner_of[pairs.expert]
        if (owner < 0).any():
            raise RuntimeError(f"layer {self.layer}: no node holds experts {sorted(set(pairs.expert[owner < 0].tolist()))}")

        out_rows = hidden_states.new_empty(pairs.row.shape[0], hidden_states.shape[1]) if self.combine_mode == EXACT else None
        result = torch.zeros_like(hidden_states) if self.combine_mode == FAST else None
        header = {
            "request_id": info.request_id if info else None,
            "token_index": info.token_index if info else None,
            "layer": self.layer,
            "combine_mode": self.combine_mode,
        }
        work = []  # (node_id, keep, needed, message, experts), in node order
        for n, node_id in enumerate(self.node_ids):
            keep = (owner == n).nonzero().squeeze(1)  # ascending: keeps Transformers' order
            if keep.numel() == 0:
                continue
            mine = pairs.select(keep)
            needed = torch.unique(mine.row)  # sorted rows: the batch order stays the same
            message = codec.encode(
                header,
                {
                    "hidden": hidden_states[needed],
                    "row": torch.searchsorted(needed, mine.row).to(torch.int32),
                    "expert": mine.expert.to(torch.int32),
                    "weight": mine.weight,
                },
            )
            work.append((node_id, keep, needed, message, torch.unique(mine.expert).tolist()))

        for (node_id, keep, needed, message, _), reply in zip(work, self._send(work)):
            self.stats.update(calls=1, bytes_sent=len(message), bytes_received=len(reply))
            _, tensors = codec.decode(reply)
            if self.combine_mode == EXACT:
                out_rows[keep] = tensors["out"]
            else:
                result.index_add_(0, needed, tensors["out"])

        if self.combine_mode == FAST:
            return result
        experts, counts = torch.unique_consecutive(pairs.expert, return_counts=True)
        return combine_exact([ExactResult(experts, counts, pairs.row, out_rows)], hidden_states)

    def _send(self, work) -> list[bytes]:
        """Send every node its message; remote ones at the same time, local ones right here."""
        if self.executor is None:
            return [self._call(node_id, message, experts) for node_id, _, _, message, experts in work]
        pending = {
            node_id: self.executor.submit(self._call, node_id, message, experts)
            for node_id, _, _, message, experts in work
            if not isinstance(self.nodes[node_id], LocalNode)
        }
        local = {
            node_id: self._call(node_id, message, experts)
            for node_id, _, _, message, experts in work
            if isinstance(self.nodes[node_id], LocalNode)
        }
        return [local[node_id] if node_id in local else pending[node_id].result() for node_id, *_ in work]

    def _call(self, node_id: str, message: bytes, experts: list[int]) -> bytes:
        try:
            return self.nodes[node_id].forward(message)
        except NodeError as exc:
            # Never skip the missing experts silently: that would quietly give worse text.
            what = "stopped responding" if exc.unreachable else "failed"
            raise NodeError(f"{node_id} {what}; experts {experts} in layer {self.layer} are unavailable ({exc})",
                            node_id, exc.unreachable) from exc
