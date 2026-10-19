"""Read only the chosen experts from the model's safetensors weight files.

A safetensors file is a small header (tensor names, shapes, byte positions) followed by the raw
numbers. `safe_open` reads the header only; `get_slice(name)[a:b]` then reads just the bytes of
sheets a..b-1 of one tensor. So a node holding 4 of 32 experts reads about 4/32 of the expert
bytes, and never the whole file.

Large models are split into several files ("shards") listed in model.safetensors.index.json;
WeightFiles hides that: you ask for a tensor name and it opens the right file.
"""

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig

from moe_dist.experts.slicing import ExpertSlice, check_expert_ids, make_slice
from moe_dist.models import adapter_for
from moe_dist.reference import DTYPES

SINGLE_FILE = "model.safetensors"
INDEX_FILE = "model.safetensors.index.json"


def model_dir(model_id: str, revision: str) -> Path:
    """Local folder with the model's config and weight files (downloads them if missing)."""
    patterns = ["*.json", "*.safetensors"]
    try:
        path = snapshot_download(model_id, revision=revision, allow_patterns=patterns, local_files_only=True)
    except Exception:  # not in the local cache yet
        path = snapshot_download(model_id, revision=revision, allow_patterns=patterns)
    return Path(path)


class WeightFiles:
    """All safetensors files of one model, opened lazily; tensors are looked up by name."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        if (self.folder / INDEX_FILE).exists():
            weight_map = json.loads((self.folder / INDEX_FILE).read_text())["weight_map"]
            self._file_of = {name: self.folder / file for name, file in weight_map.items()}
        elif (self.folder / SINGLE_FILE).exists():
            with safe_open(self.folder / SINGLE_FILE, "pt") as f:
                self._file_of = {name: self.folder / SINGLE_FILE for name in f.keys()}
        else:
            raise FileNotFoundError(f"no {SINGLE_FILE} or {INDEX_FILE} in {self.folder}")
        self._open = {}

    def names(self) -> list[str]:
        return list(self._file_of)

    def get_slice(self, name: str):
        """A handle that reads parts of tensor `name` on demand: handle[a:b] -> torch.Tensor."""
        if name not in self._file_of:
            raise KeyError(f"tensor {name!r} not found in the weight files in {self.folder}")
        file = self._file_of[name]
        if file not in self._open:
            self._open[file] = safe_open(file, "pt")
        return self._open[file].get_slice(name)

    def close(self):
        self._open.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _as_dtype(dtype) -> torch.dtype:
    return DTYPES[dtype] if isinstance(dtype, str) else dtype


def _own_memory(t: torch.Tensor) -> torch.Tensor:
    """Make sure `t` has its own memory.

    A safetensors slice is a view into the memory-mapped file ("mmap": the OS maps the file into
    memory and loads pages only when they are touched). Even `get_slice(name)[0:4]` sits on top of
    the whole tensor's mapping. Copying gives predictable memory use and lets the file be closed.
    """
    owns = t.is_contiguous() and t.untyped_storage().nbytes() == t.nbytes
    return t if owns else t.clone(memory_format=torch.contiguous_format)


def load_layer_experts(files: WeightFiles, adapter, layer: int, expert_ids, dtype) -> ExpertSlice:
    """Read the chosen experts of one layer, in the in-memory layout and the requested dtype."""
    ids = check_expert_ids(expert_ids, adapter.num_experts)
    gate_up, down = adapter.read_experts(files, layer, ids)
    # Same conversion from_pretrained does (file is bf16; bf16 -> fp32 is exact).
    gate_up, down = gate_up.to(_as_dtype(dtype)), down.to(_as_dtype(dtype))
    return make_slice(ids, _own_memory(gate_up), _own_memory(down), adapter.num_experts)


def load_experts(model_id: str, revision: str, expert_ids, dtype, layers: list[int] | None = None) -> dict:
    """Read the chosen experts for several layers (default: every MoE layer).

    Returns {layer: ExpertSlice}. The same expert numbers are used in every layer.
    """
    folder = model_dir(model_id, revision)
    adapter = adapter_for(AutoConfig.from_pretrained(folder))
    layers = adapter.moe_layers() if layers is None else layers
    with WeightFiles(folder) as files:
        return {layer: load_layer_experts(files, adapter, layer, expert_ids, dtype) for layer in layers}
