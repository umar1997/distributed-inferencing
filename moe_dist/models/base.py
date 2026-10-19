"""What every model adapter must provide.

An adapter is the only place that knows how one model family stores its experts in the weight
files. It turns them into the common in-memory layout (Transformers' packed `gate_up_proj` /
`down_proj`, see experts/slicing.py). Nothing else in the project looks at file tensor names.
"""

from abc import ABC, abstractmethod

import torch
from transformers.activations import ACT2FN


class ModelAdapter(ABC):
    def __init__(self, config):
        """`config` is the model's Transformers config (AutoConfig)."""
        self.config = config

    @property
    def num_layers(self) -> int:
        return self.config.num_hidden_layers

    @property
    @abstractmethod
    def num_experts(self) -> int:
        """Experts per MoE layer."""

    def act_fn(self):
        """The experts' activation function (SiLU for Granite and Qwen), as Transformers builds it."""
        return ACT2FN[self.config.hidden_act]

    @abstractmethod
    def moe_layers(self) -> list[int]:
        """Layer numbers that have experts (all of them for Granite; some models skip layers)."""

    @abstractmethod
    def experts_module_path(self, layer: int) -> str:
        """Where the layer's `experts` module sits in the loaded Transformers model,
        e.g. 'model.layers.3.block_sparse_moe.experts'. The dispatcher replaces this module."""

    @abstractmethod
    def read_experts(self, files, layer: int, expert_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Read only `expert_ids` (ascending, already checked) of one layer from the weight files.

        `files` is a loader.WeightFiles. Returns (gate_up_proj, down_proj) in the in-memory layout,
        in the file's own dtype; the loader converts the dtype afterwards.
        """


def expert_runs(expert_ids: list[int]) -> list[tuple[int, int]]:
    """Group ascending expert numbers into runs of neighbours, as (start, stop) ranges.

    [0, 1, 2, 8, 9, 12] -> [(0, 3), (8, 10), (12, 13)]. Reading a run is one slice of the file.
    """
    runs = []
    for e in expert_ids:
        if runs and runs[-1][1] == e:
            runs[-1] = (runs[-1][0], e + 1)
        else:
            runs.append((e, e + 1))
    return runs
