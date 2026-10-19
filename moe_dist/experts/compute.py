"""Run a node's experts on tokens, with exactly the same maths as Transformers.

Transformers' "eager" expert loop (identical for Granite, Qwen1.5-MoE and Qwen3-MoE in 5.20.1):

    for each expert e that was picked, in ascending order:
        top_k_pos, token_idx = torch.where(expert_mask[e])        # rows that picked e
        gate, up = linear(hidden[token_idx], gate_up_proj[e]).chunk(2)
        rows = linear(act(gate) * up, down_proj[e]) * top_k_weights[token_idx, top_k_pos]
        out.index_add_(0, token_idx, rows)                         # add into the token's answer

`run_experts` does the first three lines for the experts one node holds. What it returns depends
on the combine mode:

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
    if mode not in MODES:
        raise ValueError(f"unknown combine mode {mode!r}; use one of {MODES}")
    hit_ids, counts, token_parts, row_parts = [], [], [], []
    with torch.no_grad():
        for local, e in enumerate(part.expert_ids):
            # (k, T) mask, the same layout as Transformers' expert_mask[e].
            top_k_pos, token_idx = torch.where((top_k_index == e).T)
            if token_idx.numel() == 0:
                continue
            gate, up = nn.functional.linear(hidden_states[token_idx], part.gate_up_proj[local]).chunk(2, dim=-1)
            rows = nn.functional.linear(act_fn(gate) * up, part.down_proj[local])
            rows = rows * top_k_weights[token_idx, top_k_pos, None]
            hit_ids.append(e)
            counts.append(token_idx.numel())
            token_parts.append(token_idx)
            row_parts.append(rows.to(hidden_states.dtype))

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
