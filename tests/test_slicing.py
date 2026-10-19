"""Phase 1.1: cutting experts out of packed tensors (no model needed)."""

import pytest
import torch

from moe_dist.experts.slicing import NOT_HERE, check_expert_ids, local_index_table, make_slice, slice_experts

# Small stand-in shapes: 32 experts like Granite, tiny hidden/intermediate sizes.
E, H, I = 32, 16, 8


def packed(dtype):
    g = torch.Generator().manual_seed(0)
    gate_up = torch.randn(E, 2 * I, H, generator=g).to(dtype)
    down = torch.randn(E, H, I, generator=g).to(dtype)
    return gate_up, down


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ids", [[3, 7, 19], [0, 4, 8, 12], [31], list(range(8, 16))])
def test_each_cut_expert_equals_its_original_sheet(dtype, ids):
    gate_up, down = packed(dtype)
    part = slice_experts(gate_up, down, ids)
    assert part.expert_ids == ids
    assert part.gate_up_proj.shape == (len(ids), 2 * I, H)
    assert part.down_proj.shape == (len(ids), H, I)
    assert part.gate_up_proj.dtype == dtype
    for local, e in enumerate(ids):
        assert torch.equal(part.gate_up_proj[local], gate_up[e])
        assert torch.equal(part.down_proj[local], down[e])


def test_unsorted_input_is_stored_in_ascending_order():
    gate_up, down = packed(torch.float32)
    part = slice_experts(gate_up, down, [19, 3, 7])
    assert part.expert_ids == [3, 7, 19]
    assert torch.equal(part.gate_up_proj[0], gate_up[3])
    assert torch.equal(part.gate_up_proj[2], gate_up[19])


def test_cutting_all_experts_gives_the_original_back():
    gate_up, down = packed(torch.float32)
    part = slice_experts(gate_up, down, list(reversed(range(E))))
    assert torch.equal(part.gate_up_proj, gate_up)
    assert torch.equal(part.down_proj, down)


def test_cut_is_a_copy_not_a_view():
    """The full tensors must be freeable after cutting, so the cut must own its memory."""
    gate_up, down = packed(torch.float32)
    part = slice_experts(gate_up, down, [0, 1])
    assert part.gate_up_proj.untyped_storage().data_ptr() != gate_up.untyped_storage().data_ptr()
    assert part.gate_up_proj.untyped_storage().nbytes() == part.gate_up_proj.nbytes
    gate_up.zero_()
    assert part.gate_up_proj.abs().sum() > 0


def test_local_index_table():
    table = local_index_table([12, 0, 8, 4], E)
    assert table.shape == (E,)
    assert table.dtype == torch.long
    assert table[0] == 0 and table[4] == 1 and table[8] == 2 and table[12] == 3
    others = [e for e in range(E) if e not in (0, 4, 8, 12)]
    assert (table[others] == NOT_HERE).all()
    # A whole batch of router picks converts in one step.
    picks = torch.tensor([[0, 5], [12, 8]])
    assert table[picks].tolist() == [[0, NOT_HERE], [3, 2]]


def test_slice_carries_its_local_index_table():
    gate_up, down = packed(torch.float32)
    part = slice_experts(gate_up, down, [3, 7, 19])
    assert part.num_experts_total == E
    for e in [3, 7, 19]:
        assert torch.equal(part.gate_up_proj[part.local_index[e]], gate_up[e])


@pytest.mark.parametrize(
    "ids, error, message",
    [
        ([], ValueError, "empty"),
        ([1, 2, 2, 5, 5], ValueError, r"more than once: \[2, 5\]"),
        ([0, 32], ValueError, r"out of range 0\.\.31: \[32\]"),
        ([-1, 3], ValueError, r"out of range 0\.\.31: \[-1\]"),
        ([1.0], TypeError, "integers"),
        ([True], TypeError, "integers"),
    ],
)
def test_bad_expert_lists_give_clear_errors(ids, error, message):
    with pytest.raises(error, match=message):
        check_expert_ids(ids, E)
    gate_up, down = packed(torch.float32)
    with pytest.raises(error, match=message):
        slice_experts(gate_up, down, ids)


def test_mismatched_tensors_are_refused():
    gate_up, down = packed(torch.float32)
    with pytest.raises(ValueError, match="sheets"):
        make_slice([0, 1, 2], gate_up[:2], down[:3], E)
    with pytest.raises(ValueError, match="does not match"):
        make_slice([0], gate_up[:1, :, :4], down[:1], E)
    with pytest.raises(ValueError, match="experts but down has"):
        slice_experts(gate_up, down[:10], [0])
