"""Phase 2.1: the master builds the model without any expert weights."""

import pytest
import torch
from transformers import AutoConfig

from moe_dist.experts.loader import model_dir
from moe_dist.master.model_loader import ExpertsStandIn, load_shared_model
from moe_dist.models import adapter_for
from moe_dist.models.catalog import DEFAULT_MODEL, pinned_revision
from moe_dist.reference import load_reference

REVISION = pinned_revision(DEFAULT_MODEL)


@pytest.mark.model
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_shared_parts_identical_and_no_experts(dtype):
    adapter = adapter_for(AutoConfig.from_pretrained(model_dir(DEFAULT_MODEL, REVISION)))
    lean, _ = load_shared_model(DEFAULT_MODEL, REVISION, dtype, adapter)
    full, _ = load_reference(DEFAULT_MODEL, REVISION, dtype)

    lean_params = dict(lean.named_parameters())
    full_params = dict(full.named_parameters())
    experts = {n for n in full_params if ".experts." in n}
    assert len(experts) == 48  # gate_up_proj + down_proj in 24 layers
    assert set(lean_params) == set(full_params) - experts
    for name, p in lean_params.items():
        assert p.device.type == "cpu" and p.dtype == full_params[name].dtype
        assert torch.equal(p, full_params[name]), name
    # Buffers too (e.g. the rotary position table, which is computed, not loaded).
    full_buffers = dict(full.named_buffers())
    for name, b in lean.named_buffers():
        assert torch.equal(b, full_buffers[name]), name
    # Output layer still shares its weights with the embeddings ("tied"), as in the original.
    assert lean.lm_head.weight.data_ptr() == lean.model.embed_tokens.weight.data_ptr()

    for layer in adapter.moe_layers():
        assert isinstance(lean.get_submodule(adapter.experts_module_path(layer)), ExpertsStandIn)
    with pytest.raises(RuntimeError, match="stand-in was called"):
        lean.model.layers[0].block_sparse_moe.experts(None, None, None)

    lean_mb = sum(p.nbytes for p in lean.parameters()) / 2**20
    full_mb = sum(p.nbytes for p in full.parameters()) / 2**20
    print(f"\n{dtype}: shared parts {lean_mb:.0f} MB, full model {full_mb:.0f} MB")
    assert lean_mb < 0.15 * full_mb


@pytest.mark.model
def test_experts_class_is_restored_after_loading():
    """The stand-in is only used while loading; a normal load afterwards has real experts."""
    adapter = adapter_for(AutoConfig.from_pretrained(model_dir(DEFAULT_MODEL, REVISION)))
    load_shared_model(DEFAULT_MODEL, REVISION, "bf16", adapter)
    full, _ = load_reference(DEFAULT_MODEL, REVISION, "bf16")
    assert not isinstance(full.model.layers[0].block_sparse_moe.experts, ExpertsStandIn)
