"""Granite-MoE (granite-3.0-1b-a400m and relatives).

On disk the experts are already packed, under different names than in memory:

    file:    model.layers.N.block_sparse_moe.input_linear.weight   (E, 2*I, H)  gate first, then up
             model.layers.N.block_sparse_moe.output_linear.weight  (E, H, I)
    memory:  model.layers.N.block_sparse_moe.experts.gate_up_proj
             model.layers.N.block_sparse_moe.experts.down_proj

Transformers 5.20.1 only renames these while loading (transformers/conversion_mapping.py,
"granitemoe"); the numbers and their order stay the same. tests/test_loader.py checks this.
"""

import torch

from moe_dist.models.base import ModelAdapter, expert_runs


class GraniteAdapter(ModelAdapter):
    @property
    def num_experts(self) -> int:
        return self.config.num_local_experts

    def moe_layers(self) -> list[int]:
        return list(range(self.num_layers))

    def experts_module_path(self, layer: int) -> str:
        return f"model.layers.{layer}.block_sparse_moe.experts"

    def file_tensor_names(self, layer: int) -> tuple[str, str]:
        prefix = f"model.layers.{layer}.block_sparse_moe"
        return f"{prefix}.input_linear.weight", f"{prefix}.output_linear.weight"

    def read_experts(self, files, layer: int, expert_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        gate_up_name, down_name = self.file_tensor_names(layer)
        return (
            _read_sheets(files, gate_up_name, expert_ids),
            _read_sheets(files, down_name, expert_ids),
        )


def _read_sheets(files, name: str, expert_ids: list[int]) -> torch.Tensor:
    """Read only the chosen sheets of one packed tensor, one file slice per run of neighbours."""
    part = files.get_slice(name)
    pieces = [part[start:stop] for start, stop in expert_runs(expert_ids)]
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces)
