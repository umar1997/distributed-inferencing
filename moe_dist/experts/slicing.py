"""Cut a chosen set of experts out of the packed expert tensors.

In memory, every MoE layer stores its experts as two packed 3-D tensors ("a stack of sheets",
one sheet per expert), in Transformers' layout:

    gate_up_proj  (E, 2*I, H)   gate rows first, then up rows
    down_proj     (E, H, I)

A node only holds some experts, e.g. [0, 4, 8, 12]. It keeps them in a smaller stack of the same
layout, so expert 8 is sheet 2 of the smaller stack. The "local index table" converts a global
expert number (0..E-1) into its position in the smaller stack.
"""

from dataclasses import dataclass

import torch

# Marks "this node does not hold that expert" in the local index table.
NOT_HERE = -1


def check_expert_ids(expert_ids, num_experts: int) -> list[int]:
    """Validate a list of expert numbers and return it sorted (ascending).

    Sorted order is the rule everywhere: local position i is the i-th smallest expert number.
    That keeps the smaller stack in the same order as the full one.
    """
    ids = list(expert_ids)
    if not ids:
        raise ValueError("expert list is empty; a node must hold at least one expert")
    for e in ids:
        if isinstance(e, bool) or not isinstance(e, int):
            raise TypeError(f"expert numbers must be integers, got {e!r} ({type(e).__name__})")
    repeated = sorted({e for e in ids if ids.count(e) > 1})
    if repeated:
        raise ValueError(f"expert numbers listed more than once: {repeated}")
    outside = sorted(e for e in ids if not 0 <= e < num_experts)
    if outside:
        raise ValueError(f"expert numbers out of range 0..{num_experts - 1}: {outside}")
    return sorted(ids)


def local_index_table(expert_ids: list[int], num_experts: int) -> torch.Tensor:
    """Global expert number -> local position, as a lookup tensor of length E.

    table[e] is the position of expert e in the smaller stack, or NOT_HERE (-1).
    A tensor (not a dict) so a whole batch of router picks can be converted in one step:
    `table[top_k_index]`.
    """
    ids = check_expert_ids(expert_ids, num_experts)
    table = torch.full((num_experts,), NOT_HERE, dtype=torch.long)
    table[torch.tensor(ids)] = torch.arange(len(ids))
    return table


@dataclass
class ExpertSlice:
    """The experts one node holds for one layer, in Transformers' packed layout."""

    expert_ids: list[int]  # global numbers, ascending; expert_ids[i] is sheet i
    gate_up_proj: torch.Tensor  # (len(expert_ids), 2*I, H)
    down_proj: torch.Tensor  # (len(expert_ids), H, I)
    local_index: torch.Tensor  # (E,) global number -> sheet, or NOT_HERE

    @property
    def num_experts_total(self) -> int:
        return self.local_index.shape[0]

    def nbytes(self) -> int:
        return self.gate_up_proj.nbytes + self.down_proj.nbytes


def make_slice(expert_ids, gate_up_proj: torch.Tensor, down_proj: torch.Tensor, num_experts: int) -> ExpertSlice:
    """Wrap already-cut tensors (sheet i = i-th smallest expert) and check their shapes agree."""
    ids = check_expert_ids(expert_ids, num_experts)
    if gate_up_proj.dim() != 3 or down_proj.dim() != 3:
        raise ValueError(
            f"packed expert tensors must be 3-D, got gate_up {tuple(gate_up_proj.shape)}, down {tuple(down_proj.shape)}"
        )
    if gate_up_proj.shape[0] != len(ids) or down_proj.shape[0] != len(ids):
        raise ValueError(
            f"{len(ids)} experts but gate_up has {gate_up_proj.shape[0]} sheets and down has {down_proj.shape[0]}"
        )
    hidden, inter = down_proj.shape[1], down_proj.shape[2]
    if gate_up_proj.shape[1:] != (2 * inter, hidden):
        raise ValueError(
            f"gate_up {tuple(gate_up_proj.shape)} does not match down {tuple(down_proj.shape)}: "
            f"expected (n, {2 * inter}, {hidden})"
        )
    return ExpertSlice(ids, gate_up_proj, down_proj, local_index_table(ids, num_experts))


def slice_experts(gate_up_proj: torch.Tensor, down_proj: torch.Tensor, expert_ids) -> ExpertSlice:
    """Cut the chosen experts out of the full packed tensors of one layer.

    The result is a copy with its own memory (not a view), so the full tensors can be freed
    afterwards and only the chosen experts stay in memory.
    """
    num_experts = gate_up_proj.shape[0]
    if down_proj.shape[0] != num_experts:
        raise ValueError(f"gate_up has {num_experts} experts but down has {down_proj.shape[0]}")
    ids = check_expert_ids(expert_ids, num_experts)
    rows = torch.tensor(ids, device=gate_up_proj.device)
    with torch.no_grad():
        gate_up = gate_up_proj.index_select(0, rows)
        down = down_proj.index_select(0, rows)
    return make_slice(ids, gate_up, down, num_experts)
