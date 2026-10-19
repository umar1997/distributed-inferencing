"""Phase 1.3: the model with dispatchers and virtual nodes gives the same tokens as plain Transformers.

Every expert call goes through the binary message format (pack -> node -> pack -> unpack), so
the format is tested on real traffic before any network exists.
"""

import json
from pathlib import Path

import pytest
from transformers import AutoConfig

from moe_dist.experts.dispatcher import ExpertDispatcher, request_scope
from moe_dist.experts.loader import model_dir
from moe_dist.experts.placement import CoverageError
from moe_dist.master.model_loader import build_local_cluster, check_all_replaced, make_placement
from moe_dist.models import adapter_for
from moe_dist.models.catalog import DEFAULT_MODEL, pinned_revision, short_name
from moe_dist.reference import NEAR_TIE, greedy_generate, match_reference

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "tests" / "golden" / short_name(DEFAULT_MODEL)
REVISION = pinned_revision(DEFAULT_MODEL)

PLACEMENTS = {
    "three ranges": {"master": "0-9", "worker-1": "10-20", "worker-2": "21-31"},
    "interleaved": {"master": "0,3,6,9,12,15,18,21,24,27,30", "worker-1": "1,4,7,10,13,16,19,22,25,28,31",
                    "worker-2": "2,5,8,11,14,17,20,23,26,29"},
    "one node": {"master": "none", "worker-1": "0-31"},
    "one expert per node": {f"node-{e}": str(e) for e in range(32)},
}


def golden(dtype):
    return json.loads((GOLDEN_DIR / f"{dtype}.json").read_text())


def run_prompts(model, tokenizer, saved_prompts, max_new_tokens):
    results = {}
    for saved in saved_prompts:
        with request_scope(saved["id"]):
            results[saved["id"]] = greedy_generate(model, tokenizer, saved["messages"], max_new_tokens)
    return results


@pytest.mark.model
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_exact_mode_gives_identical_tokens(dtype, placement):
    """The main placement runs every golden prompt; the others run the first four (9-64 tokens)."""
    g = golden(dtype)
    prompts = g["prompts"] if placement == "three ranges" else g["prompts"][:4]
    model, tokenizer, nodes, _ = build_local_cluster(DEFAULT_MODEL, REVISION, dtype, PLACEMENTS[placement], "exact")
    results = run_prompts(model, tokenizer, prompts, g["max_new_tokens"])
    for saved in prompts:
        fresh = results[saved["id"]]
        assert fresh["prompt_tokens"] == saved["prompt_tokens"]
        verdict, at = match_reference(saved, fresh["output_tokens"], max_gap=0)
        assert verdict == "identical", f"{saved['id']}: first difference at token {at}"
    assert all(node.calls > 0 for node in nodes.values()), "every node should have been used"


# Finding (Phase 1.3): in bf16, fast mode already moves the first-step scores by 0.25-0.69, far more
# than the 0.1 near-tie limit (bf16 scores here come in steps of 0.0625). So fast + bf16 changes the
# text now and then (2 of 10 prompts). Kept as an expected failure so the gap stays visible.
FAST_BF16 = pytest.mark.xfail(reason="bf16 fast mode moves scores by more than the 0.1 near-tie limit", strict=False)


@pytest.mark.model
@pytest.mark.parametrize("dtype", ["fp32", pytest.param("bf16", marks=FAST_BF16)])
def test_fast_mode_follows_the_matching_rule(dtype):
    """Fast mode adds in a different order: identical, or different only at a near-tie."""
    g = golden(dtype)
    model, tokenizer, _, _ = build_local_cluster(DEFAULT_MODEL, REVISION, dtype, PLACEMENTS["three ranges"], "fast")
    results = run_prompts(model, tokenizer, g["prompts"], g["max_new_tokens"])
    verdicts = {s["id"]: match_reference(s, results[s["id"]]["output_tokens"], NEAR_TIE[dtype]) for s in g["prompts"]}
    print(f"\nfast mode, {dtype}: {verdicts}")
    assert not [p for p, (v, _) in verdicts.items() if v == "different"], verdicts


@pytest.mark.model
def test_every_experts_module_is_replaced():
    model, _, _, dispatchers = build_local_cluster(DEFAULT_MODEL, REVISION, "bf16", PLACEMENTS["three ranges"])
    assert sorted(dispatchers) == list(range(24))
    assert sum(isinstance(m, ExpertDispatcher) for m in model.modules()) == 24
    check_all_replaced(model, adapter_for(model.config))
    # And the check notices a module we missed:
    layer3 = model.get_submodule("model.layers.3.block_sparse_moe")
    layer3.experts = layer3.router  # anything that is not a dispatcher
    with pytest.raises(RuntimeError, match="layer 3: experts module was not replaced"):
        check_all_replaced(model, adapter_for(model.config))


@pytest.mark.model
def test_messages_carry_request_and_token_numbers():
    model, tokenizer, nodes, _ = build_local_cluster(DEFAULT_MODEL, REVISION, "bf16", {"only": "0-31"})
    seen = []
    node = nodes["only"]
    original = node.forward

    def spy(message):
        from moe_dist.transport.codec import decode

        header, _ = decode(message)
        seen.append((header["request_id"], header["token_index"], header["layer"]))
        return original(message)

    node.forward = spy
    with request_scope("req-42"):
        greedy_generate(model, tokenizer, [{"role": "user", "content": "Hi"}], max_new_tokens=3)
    assert {r for r, _, _ in seen} == {"req-42"}
    assert sorted({t for _, t, _ in seen}) == [0, 1, 2]  # the prompt, then 2 more passes
    assert [layer for _, t, layer in seen if t == 0] == list(range(24))


def test_missing_expert_is_refused_before_loading():
    adapter = adapter_for(AutoConfig.from_pretrained(model_dir(DEFAULT_MODEL, REVISION)))
    nodes = dict(PLACEMENTS["three ranges"], **{"worker-1": "10-16,18-20"})
    with pytest.raises(CoverageError, match=r"missing experts: \[17\] in all layers"):
        make_placement(adapter, nodes)
    with pytest.raises(CoverageError, match=r"missing experts: \[17\] in all layers"):
        build_local_cluster(DEFAULT_MODEL, REVISION, "bf16", nodes)
