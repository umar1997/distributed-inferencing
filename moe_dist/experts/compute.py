"""Run a node's experts on tokens, with exactly the same maths as Transformers.

Transformers' "eager" expert loop (identical for Granite, Qwen1.5-MoE and Qwen3-MoE in 5.20.1):

    for each expert e that was picked, in ascending order:
        top_k_pos, token_idx = torch.where(expert_mask[e])        # rows that picked e
        gate, up = linear(hidden[token_idx], gate_up_proj[e]).chunk(2)
        rows = linear(act(gate) * up, down_proj[e]) * top_k_weights[token_idx, top_k_pos]
        out.index_add_(0, token_idx, rows)                         # add into the token's answer

`plan_pairs` turns the router's picks into a work list of (token, expert, weight) pairs in that
same order; `run_pairs` does the first three lines for one node's part of the list (this is what
a worker runs); `run_experts` does both in one step. What comes back depends on the combine mode:

- exact: the weighted rows for every (token, expert) pair, not yet added up. The master adds the
  rows of all nodes in ascending expert order (`combine_exact`), which is the same order as
  Transformers, so the result is bit-identical (same bits, not just close).
- fast: each node adds up its own experts per token first and sends one row per token
  (`combine_fast` then adds the nodes' sums). Less data to send, but the additions happen in a
  different order, so the result can differ in the last digits.

Bit-identical also needs every expert to see exactly the same rows in the same order as in
Transformers, because a matrix multiplication may round differently for a different batch of
rows. `torch.where` on the (top_k, tokens) mask gives that order: all tokens that picked e as
their 1st choice, then as their 2nd choice, and so on. If `hidden_states` is only a subset of the
batch, it must keep the batch order (the order of `torch.where` is then unchanged).
"""

from dataclasses import dataclass

import torch
from torch import nn

from moe_dist.experts.slicing import ExpertSlice

EXACT = "exact"
FAST = "fast"
MODES = (EXACT, FAST)


@dataclass
class ExactResult:
    """Weighted rows for every (token, expert) pair a node computed, grouped by expert.

    Rows of expert `expert_ids[i]` are the next `counts[i]` rows of `rows` (and of `token_idx`).
    """

    expert_ids: torch.Tensor  # (n_experts_hit,) long, ascending
    counts: torch.Tensor  # (n_experts_hit,) long
    token_idx: torch.Tensor  # (n_pairs,) long, row number in hidden_states
    rows: torch.Tensor  # (n_pairs, H), already multiplied by the router weight


@dataclass
class FastResult:
    """One row per token: the sum of this node's experts for that token (zeros if none)."""

    sums: torch.Tensor  # (T, H)


@dataclass
class Pairs:
    """The work list: one entry per (token, expert) pair, in Transformers' order.

    Sorted by expert number; inside one expert, by choice rank (1st choice, 2nd choice, ...) and
    then by row. This is exactly the order of `torch.where(expert_mask[e])` in Transformers.
    """

    row: torch.Tensor  # (n,) long, row number in hidden_states
    expert: torch.Tensor  # (n,) long, global expert number
    weight: torch.Tensor  # (n,) router weight, same dtype as hidden_states

    def select(self, keep: torch.Tensor) -> "Pairs":
        """A subset (keep = positions, ascending), still in the same order."""
        return Pairs(self.row[keep], self.expert[keep], self.weight[keep])


def plan_pairs(top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> Pairs:
    """Router picks (T, k) -> the work list for all experts, in Transformers' order."""
    num_tokens = top_k_index.shape[0]
    # Position p = rank * T + row, the row-major order of Transformers' (k, T) mask.
    flat_expert = top_k_index.T.reshape(-1)
    order = torch.sort(flat_expert, stable=True).indices  # stable: keeps p ascending per expert
    row, rank = order % num_tokens, order // num_tokens
    return Pairs(row=row, expert=flat_expert[order], weight=top_k_weights[row, rank])


def run_pairs(part: ExpertSlice, act_fn, hidden_states: torch.Tensor, pairs: Pairs, mode: str = EXACT):
    """Run a work list on this node's experts. This is what a worker does for one request.

    hidden_states (R, H) holds the rows the pairs point to. In exact mode the result rows are in
    the same order as `pairs`.
    """
    if mode not in MODES:
        raise ValueError(f"unknown combine mode {mode!r}; use one of {MODES}")
    if pairs.expert.numel() and (pairs.expert[1:] < pairs.expert[:-1]).any():
        raise ValueError("pairs must be sorted by expert number")
    local_of = part.local_index[pairs.expert]
    if (local_of < 0).any():
        missing = sorted(set(pairs.expert[local_of < 0].tolist()))
        raise ValueError(f"experts {missing} are not on this node (it holds {part.expert_ids})")

    hit_ids, counts, token_parts, row_parts = [], [], [], []
    experts, sizes = torch.unique_consecutive(pairs.expert, return_counts=True)
    start = 0
    with torch.no_grad():
        for e, n in zip(experts.tolist(), sizes.tolist()):
            local = part.local_index[e].item()
            token_idx = pairs.row[start : start + n]
            gate, up = nn.functional.linear(hidden_states[token_idx], part.gate_up_proj[local]).chunk(2, dim=-1)
            rows = nn.functional.linear(act_fn(gate) * up, part.down_proj[local])
            rows = rows * pairs.weight[start : start + n, None]
            hit_ids.append(e)
            counts.append(n)
            token_parts.append(token_idx)
            row_parts.append(rows.to(hidden_states.dtype))
            start += n
    return _result(hidden_states, hit_ids, counts, token_parts, row_parts, mode)


def run_experts(
    part: ExpertSlice,
    act_fn,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    mode: str = EXACT,
) -> ExactResult | FastResult:
    """Run the experts in `part` for the tokens that picked them.

    hidden_states (T, H); top_k_index (T, k) global expert numbers; top_k_weights (T, k).
    Picks of experts this node does not hold are ignored (another node computes them).
    """
    pairs = plan_pairs(top_k_index, top_k_weights)
    mine = (part.local_index[pairs.expert] >= 0).nonzero().squeeze(1)
    return run_pairs(part, act_fn, hidden_states, pairs.select(mine), mode)


def _result(hidden_states, hit_ids, counts, token_parts, row_parts, mode):
    if mode == FAST:
        sums = torch.zeros_like(hidden_states)
        for token_idx, rows in zip(token_parts, row_parts):
            sums.index_add_(0, token_idx, rows)
        return FastResult(sums)

    hidden = hidden_states.shape[1]
    return ExactResult(
        expert_ids=torch.tensor(hit_ids, dtype=torch.long),
        counts=torch.tensor(counts, dtype=torch.long),
        token_idx=torch.cat(token_parts) if token_parts else torch.empty(0, dtype=torch.long),
        rows=torch.cat(row_parts) if row_parts else hidden_states.new_empty(0, hidden),
    )


def combine_exact(results: list[ExactResult], like: torch.Tensor) -> torch.Tensor:
    """Add every node's rows into one answer per token, in ascending expert order.

    `like` is the layer's hidden_states (gives shape, dtype and device of the answer).
    Raises if two nodes sent the same expert (each expert must be computed exactly once).
    """
    pieces = {}  # expert number -> (token_idx, rows)
    for r in results:
        starts = torch.cumsum(r.counts, 0) - r.counts
        for e, start, n in zip(r.expert_ids.tolist(), starts.tolist(), r.counts.tolist()):
            if e in pieces:
                raise ValueError(f"expert {e} was computed by more than one node")
            pieces[e] = (r.token_idx[start : start + n], r.rows[start : start + n])
    out = torch.zeros_like(like)
    for e in sorted(pieces):
        token_idx, rows = pieces[e]
        out.index_add_(0, token_idx, rows.to(out.dtype))
    return out


def combine_fast(results: list[FastResult], like: torch.Tensor) -> torch.Tensor:
    """Add the nodes' per-token sums (order of nodes as given)."""
    out = torch.zeros_like(like)
    for r in results:
        out += r.sums
    return out
