"""Build the model with dispatchers in place of every layer's `experts` module.

The master never holds the full model: `load_shared_model` lets Transformers load everything
except the experts (attention, embeddings, routers, norms), with Transformers' own renames and
conversions. Each MoE layer gets an empty stand-in for its experts, and the stand-ins are then
swapped for ExpertDispatchers. The experts themselves live on the nodes, which read only their
own experts from the weight file (experts/loader.py).

Everything else (attention, router, KV cache, chat template, `generate()`) stays Transformers' code.
"""

from contextlib import contextmanager

from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from moe_dist.experts.compute import EXACT
from moe_dist.experts.dispatcher import ExpertDispatcher, LocalNode
from moe_dist.experts.loader import load_experts, model_dir
from moe_dist.experts.placement import Placement, parse_expert_ids
from moe_dist.models import adapter_for
from moe_dist.reference import DTYPES


class ExpertsStandIn(nn.Module):
    """Takes the place of a layer's experts while loading: no weights, so none are loaded."""

    def __init__(self, layer_config=None):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise RuntimeError("experts stand-in was called: install_dispatchers() was not run")


@contextmanager
def _experts_left_out(block_class):
    """While active, every new MoE block gets an ExpertsStandIn instead of real experts.

    Transformers first builds the model on the "meta" device (shapes only, no memory), so the
    real experts created by the original __init__ never take memory; they are replaced at once.
    """
    original_init = block_class.__init__

    def init_without_experts(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.experts = ExpertsStandIn()

    block_class.__init__ = init_without_experts
    try:
        yield
    finally:
        block_class.__init__ = original_init


def load_shared_model(model_id: str, revision: str, dtype: str, adapter):
    """The model without any expert weights, plus its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, clean_up_tokenization_spaces=False)
    verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()  # its load report lists the skipped experts; we check that ourselves below
    try:
        with _experts_left_out(adapter.moe_block_class()):
            model, info = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, dtype=DTYPES[dtype], output_loading_info=True
            )
    finally:
        hf_logging.set_verbosity(verbosity)
    expert_paths = tuple(f"{adapter.experts_module_path(layer)}." for layer in adapter.moe_layers())
    not_experts = [k for k in info["unexpected_keys"] if not k.startswith(expert_paths)]
    if info["missing_keys"] or not_experts or info.get("mismatched_keys"):
        raise RuntimeError(
            f"loading the shared parts went wrong: missing {sorted(info['missing_keys'])[:5]}, "
            f"unexpected {not_experts[:5]}, mismatched {list(info.get('mismatched_keys', []))[:5]}"
        )
    model.eval()
    return model, tokenizer


def make_placement(adapter, node_experts: dict) -> Placement:
    """{node_id: '0-10' | [0, 1, ...] | 'none'} -> Placement (same experts in every layer), coverage checked."""
    parsed = {
        node_id: parse_expert_ids(ids, adapter.num_experts) if isinstance(ids, str) else list(ids)
        for node_id, ids in node_experts.items()
    }
    placement = Placement.same_in_every_layer(parsed, adapter.moe_layers(), adapter.num_experts)
    placement.check_coverage()
    return placement


def install_dispatchers(model, adapter, placement: Placement, nodes: dict, combine_mode: str = EXACT, executor=None):
    """Swap every MoE layer's `experts` module for a dispatcher; returns {layer: dispatcher}."""
    dispatchers = {}
    for i, layer in enumerate(adapter.moe_layers()):
        parent_path, _, name = adapter.experts_module_path(layer).rpartition(".")
        dispatcher = ExpertDispatcher(layer, placement, nodes, combine_mode, first_layer=i == 0, executor=executor)
        setattr(model.get_submodule(parent_path), name, dispatcher)
        dispatchers[layer] = dispatcher
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


def local_node(node_id: str, model_id: str, revision: str, dtype: str, adapter, expert_ids) -> LocalNode:
    """A node in this program holding `expert_ids` in every MoE layer (read from the file)."""
    return LocalNode(node_id, load_experts(model_id, revision, expert_ids, dtype), adapter.act_fn())


def build_local_cluster(model_id: str, revision: str, dtype: str, node_experts: dict, combine_mode: str = EXACT):
    """Model + virtual nodes in one program. Returns (model, tokenizer, nodes, dispatchers)."""
    adapter = adapter_for(AutoConfig.from_pretrained(model_dir(model_id, revision)))
    placement = make_placement(adapter, node_experts)  # refuses early if experts are missing
    first = adapter.moe_layers()[0]
    nodes = {
        node_id: local_node(node_id, model_id, revision, dtype, adapter, placement.experts_of(node_id, first))
        for node_id in placement.node_ids()
    }
    model, tokenizer = load_shared_model(model_id, revision, dtype, adapter)
    dispatchers = install_dispatchers(model, adapter, placement, nodes, combine_mode)
    return model, tokenizer, nodes, dispatchers
