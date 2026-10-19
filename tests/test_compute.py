"""Phase 1.2: our expert code gives exactly the same numbers as Transformers' expert code.

The reference is Transformers' own experts class (eager loop), so any change in a future
Transformers version shows up here. Written for any shapes: Granite at full size, and Qwen1.5 /
Qwen3 with their real expert counts but smaller matrices (to stay fast).
"""

import pytest
import torch
from transformers import GraniteMoeConfig, Qwen2MoeConfig, Qwen3MoeConfig
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeExperts
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeExperts
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from moe_dist.experts.compute import (
    EXACT,
    FAST,
    ExactResult,
    Pairs,
    combine_exact,
    combine_fast,
    plan_pairs,
    run_experts,
    run_pairs,
)
from moe_dist.experts.slicing import slice_experts
from moe_dist.models.catalog import DEFAULT_MODEL, pinned_revision
from moe_dist.reference import build_inputs, load_reference


def granite():
    return GraniteMoeExperts(
        GraniteMoeConfig(hidden_size=1024, intermediate_size=512, num_local_experts=32, num_experts_per_tok=8)
    ), 8


def qwen15_small():
    cfg = Qwen2MoeConfig(hidden_size=256, moe_intermediate_size=128, num_experts=60, num_experts_per_tok=4)
    return Qwen2MoeExperts(cfg), 4


def qwen3_small():
    cfg = Qwen3MoeConfig(hidden_size=256, moe_intermediate_size=96, num_experts=128, num_experts_per_tok=8)
    return Qwen3MoeExperts(cfg), 8


FAMILIES = {"granite": granite, "qwen1.5": qwen15_small, "qwen3": qwen3_small}
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def make_experts(family, dtype):
    module, top_k = FAMILIES[family]()
    module.config._experts_implementation = "eager"
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        module.gate_up_proj.copy_(torch.randn(module.gate_up_proj.shape, generator=g) * 0.02)
        module.down_proj.copy_(torch.randn(module.down_proj.shape, generator=g) * 0.02)
    return module.to(DTYPES[dtype]).eval(), top_k


def make_inputs(module, top_k, num_tokens, seed=2):
    """Random tokens and router picks, made the way the routers make them (top-k of scores)."""
    g = torch.Generator().manual_seed(seed)
    dtype = module.gate_up_proj.dtype
    hidden = torch.randn(num_tokens, module.hidden_dim, generator=g).to(dtype)
    scores = torch.randn(num_tokens, module.num_experts, generator=g)
    top_logits, top_k_index = scores.topk(top_k, dim=-1)
    return hidden, top_k_index, top_logits.softmax(-1).to(dtype)


def placements(num_experts):
    """Ways to split the experts between nodes (each a list of expert lists)."""
    third = num_experts // 3
    return {
        "three ranges": [list(range(0, third)), list(range(third, 2 * third)), list(range(2 * third, num_experts))],
        "interleaved": [list(range(n, num_experts, 3)) for n in range(3)],
        "one node": [list(range(num_experts))],
        "one expert per node": [[e] for e in range(num_experts)],
    }


def cut(module, nodes):
    return [slice_experts(module.gate_up_proj.data, module.down_proj.data, ids) for ids in nodes]


@torch.no_grad()
def reference(module, hidden, top_k_index, top_k_weights):
    return module(hidden, top_k_index, top_k_weights)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("num_tokens", [1, 9, 300])  # one decode step, a short prompt, a long prompt
def test_exact_mode_is_bit_identical(family, dtype, num_tokens):
    module, top_k = make_experts(family, dtype)
    hidden, idx, weights = make_inputs(module, top_k, num_tokens)
    expected = reference(module, hidden, idx, weights)
    for name, nodes in placements(module.num_experts).items():
        results = [run_experts(part, module.act_fn, hidden, idx, weights, EXACT) for part in cut(module, nodes)]
        got = combine_exact(results, hidden)
        assert got.dtype == expected.dtype
        assert torch.equal(got, expected), f"{name}: max diff {(got.float() - expected.float()).abs().max()}"


@pytest.mark.parametrize("dtype", DTYPES)
def test_exact_mode_with_only_the_needed_rows(dtype):
    """Phase 1.3 sends each node only the tokens that picked its experts. As long as those rows
    keep their batch order, the result stays bit-identical."""
    module, top_k = make_experts("granite", dtype)
    hidden, idx, weights = make_inputs(module, top_k, 40)
    expected = reference(module, hidden, idx, weights)
    results = []
    for part in cut(module, placements(module.num_experts)["three ranges"]):
        needed = torch.isin(idx, torch.tensor(part.expert_ids)).any(dim=1).nonzero().squeeze(1)  # ascending
        r = run_experts(part, module.act_fn, hidden[needed], idx[needed], weights[needed], EXACT)
        r.token_idx = needed[r.token_idx]  # back to row numbers of the whole batch
        results.append(r)
    assert torch.equal(combine_exact(results, hidden), expected)


@pytest.mark.parametrize("dtype, tolerance", [("fp32", 1e-5), ("bf16", 2e-2)])
def test_fast_mode_is_close(dtype, tolerance):
    module, top_k = make_experts("granite", dtype)
    hidden, idx, weights = make_inputs(module, top_k, 300)
    expected = reference(module, hidden, idx, weights)
    for name, nodes in placements(module.num_experts).items():
        results = [run_experts(part, module.act_fn, hidden, idx, weights, FAST) for part in cut(module, nodes)]
        got = combine_fast(results, hidden)
        if name == "one node":  # same additions in the same order -> still identical
            assert torch.equal(got, expected)
        diff = (got.float() - expected.float()).abs().max().item()
        assert diff <= tolerance * expected.float().abs().max().item(), f"{name}: max diff {diff}"


def test_node_without_picked_experts_returns_nothing():
    module, top_k = make_experts("granite", "fp32")
    hidden, idx, weights = make_inputs(module, top_k, 3)
    unused = sorted(set(range(32)) - set(idx.flatten().tolist()))
    part = slice_experts(module.gate_up_proj.data, module.down_proj.data, unused)
    r = run_experts(part, module.act_fn, hidden, idx, weights, EXACT)
    assert r.expert_ids.numel() == 0 and r.rows.shape == (0, 1024)
    assert torch.equal(combine_exact([r], hidden), torch.zeros_like(hidden))
    assert torch.equal(run_experts(part, module.act_fn, hidden, idx, weights, FAST).sums, torch.zeros_like(hidden))


def test_result_layout():
    module, top_k = make_experts("granite", "fp32")
    hidden, idx, weights = make_inputs(module, top_k, 20)
    part = slice_experts(module.gate_up_proj.data, module.down_proj.data, [3, 7, 19])
    r = run_experts(part, module.act_fn, hidden, idx, weights, EXACT)
    assert r.expert_ids.tolist() == [e for e in [3, 7, 19] if (idx == e).any()]
    assert r.counts.tolist() == [int((idx == e).sum()) for e in r.expert_ids.tolist()]
    assert r.token_idx.shape[0] == r.rows.shape[0] == int(r.counts.sum())


def test_same_expert_from_two_nodes_is_refused():
    module, top_k = make_experts("granite", "fp32")
    hidden, idx, weights = make_inputs(module, top_k, 20)
    parts = cut(module, [list(range(0, 20)), list(range(10, 32))])
    results = [run_experts(p, module.act_fn, hidden, idx, weights, EXACT) for p in parts]
    with pytest.raises(ValueError, match="more than one node"):
        combine_exact(results, hidden)


def test_unknown_mode_is_refused():
    module, top_k = make_experts("granite", "fp32")
    hidden, idx, weights = make_inputs(module, top_k, 2)
    with pytest.raises(ValueError, match="unknown combine mode"):
        run_experts(cut(module, [[0]])[0], module.act_fn, hidden, idx, weights, "approx")


def test_empty_exact_result_type():
    module, top_k = make_experts("granite", "bf16")
    hidden, idx, weights = make_inputs(module, top_k, 1)
    part = slice_experts(module.gate_up_proj.data, module.down_proj.data, [31])
    r = run_experts(part, module.act_fn, hidden, idx, weights, EXACT)
    assert isinstance(r, ExactResult)
    assert r.rows.dtype == torch.bfloat16


def test_plan_pairs_follows_transformers_order():
    """Inside one expert: by choice rank first (as in Transformers' torch.where), not by row."""
    idx = torch.tensor([[3, 7], [7, 3], [3, 1]])  # row 1 picked expert 3 as its 2nd choice
    weights = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.9, 0.1]])
    pairs = plan_pairs(idx, weights)
    assert pairs.expert.tolist() == [1, 3, 3, 3, 7, 7]
    assert pairs.row.tolist() == [2, 0, 2, 1, 1, 0]
    assert pairs.weight.tolist() == pytest.approx([0.1, 0.6, 0.9, 0.3, 0.7, 0.4])
    # The same order Transformers' loop uses:
    for e in [1, 3, 7]:
        _, token_idx = torch.where(torch.nn.functional.one_hot(idx, 32).permute(2, 1, 0)[e])
        assert pairs.row[pairs.expert == e].tolist() == token_idx.tolist()


def test_run_pairs_refuses_bad_work_lists():
    module, top_k = make_experts("granite", "fp32")
    hidden = torch.randn(2, 1024)
    part = slice_experts(module.gate_up_proj.data, module.down_proj.data, [3, 7])
    unsorted = Pairs(torch.tensor([0, 1]), torch.tensor([7, 3]), torch.ones(2))
    with pytest.raises(ValueError, match="sorted"):
        run_pairs(part, module.act_fn, hidden, unsorted)
    foreign = Pairs(torch.tensor([0, 1]), torch.tensor([3, 9]), torch.ones(2))
    with pytest.raises(ValueError, match=r"experts \[9\] are not on this node"):
        run_pairs(part, module.act_fn, hidden, foreign)


# --- With the real model: real weights and real inputs captured during a forward pass ---


@pytest.mark.model
@pytest.mark.parametrize("dtype", DTYPES)
def test_real_layers_are_bit_identical(dtype):
    model, tokenizer = load_reference(DEFAULT_MODEL, pinned_revision(DEFAULT_MODEL), dtype)
    layers = [0, 5, 11, 17, 23]
    captured = {}

    def capture(layer):
        def hook(module, args, output):
            captured.setdefault(layer, []).append((module, [a.clone() for a in args], output.clone()))

        return hook

    handles = [
        model.get_submodule(f"model.layers.{n}.block_sparse_moe.experts").register_forward_hook(capture(n))
        for n in layers
    ]
    try:
        messages = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
        with torch.no_grad():
            model.generate(**build_inputs(tokenizer, messages), max_new_tokens=3, do_sample=False)
    finally:
        for h in handles:
            h.remove()

    for n in layers:
        assert len(captured[n]) == 3  # the prompt, then one call per new token
        for module, (hidden, idx, weights), expected in captured[n]:
            nodes = placements(32)["three ranges"]
            results = [run_experts(p, module.act_fn, hidden, idx, weights, EXACT) for p in cut(module, nodes)]
            assert torch.equal(combine_exact(results, hidden), expected), f"layer {n}, {hidden.shape[0]} tokens"
