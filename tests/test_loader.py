"""Phase 1.1: reading only the chosen experts from the weight file.

The main check: experts read from the file are IDENTICAL to what from_pretrained puts in memory.
That proves the adapter's file names, the slicing and the gate/up order are all right.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

from moe_dist.experts.loader import WeightFiles, load_experts, load_layer_experts, model_dir
from moe_dist.models import adapter_for
from moe_dist.models.base import expert_runs
from moe_dist.models.catalog import DEFAULT_MODEL, pinned_revision
from moe_dist.reference import load_reference

ROOT = Path(__file__).resolve().parent.parent
REVISION = pinned_revision(DEFAULT_MODEL)
LAYERS = [0, 5, 11, 17, 23]
EXPERT_SETS = [[0, 4, 8, 12], [3, 7, 19], [30, 31], list(range(8, 16))]


def test_expert_runs():
    assert expert_runs([0, 1, 2, 8, 9, 12]) == [(0, 3), (8, 10), (12, 13)]
    assert expert_runs([5]) == [(5, 6)]
    assert expert_runs(list(range(32))) == [(0, 32)]


@pytest.fixture(scope="module")
def files():
    with WeightFiles(model_dir(DEFAULT_MODEL, REVISION)) as f:
        yield f


@pytest.fixture(scope="module")
def adapter(files):
    return adapter_for(AutoConfig.from_pretrained(files.folder))


@pytest.mark.model
def test_adapter_knows_granite(adapter, files):
    assert adapter.num_experts == 32
    assert adapter.moe_layers() == list(range(24))
    for layer in adapter.moe_layers():
        for name in adapter.file_tensor_names(layer):
            assert name in files.names()


@pytest.mark.model
def test_bad_tensor_name_gives_clear_error(files):
    with pytest.raises(KeyError, match="not found"):
        files.get_slice("model.layers.99.block_sparse_moe.input_linear.weight")


@pytest.mark.model
@pytest.mark.parametrize("ids", EXPERT_SETS)
def test_partial_read_equals_full_read(files, adapter, ids):
    """Reading some sheets gives the same numbers as reading the whole tensor and cutting it."""
    gate_up_name, down_name = adapter.file_tensor_names(11)
    full_gate_up = files.get_slice(gate_up_name)[:]
    full_down = files.get_slice(down_name)[:]
    part = load_layer_experts(files, adapter, 11, ids, torch.bfloat16)
    assert torch.equal(part.gate_up_proj, full_gate_up[ids])
    assert torch.equal(part.down_proj, full_down[ids])


@pytest.mark.model
@pytest.mark.parametrize("ids", [list(range(8, 16)), [0, 4, 8, 12], list(range(32))])
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_loaded_experts_own_their_memory(files, adapter, ids, dtype):
    """Not a view into the memory-mapped file (safetensors slices are views, even partial ones)."""
    part = load_layer_experts(files, adapter, 3, ids, dtype)
    for t in (part.gate_up_proj, part.down_proj):
        assert t.is_contiguous()
        assert t.untyped_storage().nbytes() == t.nbytes


@pytest.fixture(scope="module", params=["fp32", "bf16"])
def reference(request):
    model, _ = load_reference(DEFAULT_MODEL, REVISION, request.param)
    return request.param, model


@pytest.mark.model
def test_file_experts_identical_to_from_pretrained(reference, files, adapter):
    dtype, model = reference
    for layer in LAYERS:
        in_memory = model.get_submodule(adapter.experts_module_path(layer))
        for ids in EXPERT_SETS + [list(range(32))]:
            part = load_layer_experts(files, adapter, layer, ids, dtype)
            assert part.gate_up_proj.dtype == in_memory.gate_up_proj.dtype
            assert torch.equal(part.gate_up_proj, in_memory.gate_up_proj.data[ids]), f"layer {layer} gate_up {ids}"
            assert torch.equal(part.down_proj, in_memory.down_proj.data[ids]), f"layer {layer} down {ids}"


@pytest.mark.model
def test_load_experts_covers_every_layer():
    parts = load_experts(DEFAULT_MODEL, REVISION, [3, 7], "bf16")
    assert sorted(parts) == list(range(24))
    assert all(p.expert_ids == [3, 7] for p in parts.values())


# Run in a fresh process so its peak memory is not mixed up with the other tests.
MEMORY_SCRIPT = """
import json, os, resource, subprocess, sys
from moe_dist.experts.loader import load_experts
from moe_dist.models.catalog import DEFAULT_MODEL, pinned_revision

def peak_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10  # bytes on macOS, KB on Linux

def current_mb():
    return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout) / 2**10

ids = json.loads(sys.argv[1])
before = current_mb()
parts = load_experts(DEFAULT_MODEL, pinned_revision(DEFAULT_MODEL), ids, "bf16")
# Read every number once, so memory that only exists "on paper" would show up too.
checksum = sum(float(p.gate_up_proj.float().sum() + p.down_proj.float().sum()) for p in parts.values())
print(json.dumps({"before_mb": before, "peak_mb": peak_mb(), "kept_mb": current_mb(),
                  "tensor_mb": sum(p.nbytes() for p in parts.values()) / 2**20}))
"""


def measure(ids):
    out = subprocess.run(
        [sys.executable, "-c", MEMORY_SCRIPT, json.dumps(ids)], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.model
def test_loading_4_experts_uses_about_4_32_of_the_memory():
    four, every = measure([0, 4, 8, 12]), measure(list(range(32)))
    print(f"\n4 experts: {four}\n32 experts: {every}")
    # The kept tensors are exactly 4/32 of all expert tensors.
    assert four["tensor_mb"] == pytest.approx(every["tensor_mb"] * 4 / 32)
    # Memory kept once loading is done (file closed): about 4/32 of loading every expert.
    kept_four = four["kept_mb"] - four["before_mb"]
    kept_every = every["kept_mb"] - every["before_mb"]
    assert kept_four < 0.2 * kept_every, f"4 experts kept {kept_four:.0f} MB vs {kept_every:.0f} MB for all 32"
    # The highest point during loading is higher (the file pages being read, plus our copy),
    # but still far below reading the whole file.
    peak_four = four["peak_mb"] - four["before_mb"]
    peak_every = every["peak_mb"] - every["before_mb"]
    assert peak_four < 0.35 * peak_every, f"4 experts peaked {peak_four:.0f} MB vs {peak_every:.0f} MB for all 32"
