"""Build the model with dispatchers in place of every layer's `experts` module.

Phase 1.3: everything runs in one program. The full model is loaded with Transformers, the
"virtual nodes" read their own experts from the weight file (experts/loader.py), and then each
MoE layer's `experts` module is swapped for an ExpertDispatcher. The original expert weights are
dropped, so only the nodes' copies remain. (Phase 2.1 changes this to load only the shared parts
on the master.)

Everything else (attention, router, KV cache, chat template, `generate()`) stays Transformers' code.
"""

import gc

from transformers import AutoConfig

from moe_dist.experts.compute import EXACT
from moe_dist.experts.dispatcher import ExpertDispatcher, LocalNode
from moe_dist.experts.loader import load_experts, model_dir
from moe_dist.experts.placement import Placement, parse_expert_ids
from moe_dist.models import adapter_for
from moe_dist.reference import load_reference


def make_placement(adapter, node_experts: dict) -> Placement:
    """{node_id: '0-10' | [0, 1, ...] | 'none'} -> Placement (same experts in every layer), coverage checked."""
    parsed = {
        node_id: parse_expert_ids(ids, adapter.num_experts) if isinstance(ids, str) else list(ids)
        for node_id, ids in node_experts.items()
    }
    placement = Placement.same_in_every_layer(parsed, adapter.moe_layers(), adapter.num_experts)
    placement.check_coverage()
    return placement


def install_dispatchers(model, adapter, placement: Placement, nodes: dict, combine_mode: str = EXACT):
    """Swap every MoE layer's `experts` module for a dispatcher; returns {layer: dispatcher}."""
    dispatchers = {}
    for i, layer in enumerate(adapter.moe_layers()):
        parent_path, _, name = adapter.experts_module_path(layer).rpartition(".")
        dispatcher = ExpertDispatcher(layer, placement, nodes, combine_mode, first_layer=i == 0)
        setattr(model.get_submodule(parent_path), name, dispatcher)
        dispatchers[layer] = dispatcher
    gc.collect()  # the replaced modules held the full expert weights
    check_all_replaced(model, adapter)
    return dispatchers


def check_all_replaced(model, adapter):
    """Fail loudly if any expert weights are left in the model (an `experts` module we missed)."""
    for layer in adapter.moe_layers():
        module = model.get_submodule(adapter.experts_module_path(layer))
        if not isinstance(module, ExpertDispatcher):
            raise RuntimeError(f"layer {layer}: experts module was not replaced ({type(module).__name__})")
    leftovers = [name for name, _ in model.named_parameters() if name.endswith(("gate_up_proj", "down_proj"))]
    if leftovers:
        raise RuntimeError(f"expert weights still in the model: {leftovers[:3]}...")


def build_local_cluster(model_id: str, revision: str, dtype: str, node_experts: dict, combine_mode: str = EXACT):
    """Model + virtual nodes in one program. Returns (model, tokenizer, nodes, dispatchers)."""
    adapter = adapter_for(AutoConfig.from_pretrained(model_dir(model_id, revision)))
    placement = make_placement(adapter, node_experts)  # refuses early if experts are missing
    nodes = {
        node_id: LocalNode(
            node_id,
            load_experts(model_id, revision, placement.experts_of(node_id, adapter.moe_layers()[0]), dtype),
            adapter.act_fn(),
        )
        for node_id in placement.node_ids()
    }
    model, tokenizer = load_reference(model_id, revision, dtype)
    dispatchers = install_dispatchers(model, adapter, placement, nodes, combine_mode)
    return model, tokenizer, nodes, dispatchers
