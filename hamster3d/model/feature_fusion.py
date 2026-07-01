"""Feature fusion modules for combining 2D and 3D features."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FeatureFusionConfig:
    """Configuration for feature fusion."""
    fusion_method: str = "add"
    hidden_size: int = 3584
    num_heads: int = 8
    dropout: float = 0.1
    num_layers: int = 1


class FeatureFusionModule(nn.Module):
    """Feature fusion module — inference-only, supports resize_and_add and add methods."""

    def __init__(self, config: FeatureFusionConfig):
        super().__init__()
        self.config = config
        self.fusion_method = config.fusion_method
        self.hidden_size = config.hidden_size
        self._forward_called = False

    def forward(
        self,
        features_2d: torch.Tensor,
        features_3d: torch.Tensor,
        grid_2d: tuple[int, int] = None,
        grid_3d: tuple[int, int] = None,
    ) -> torch.Tensor:
        if self.fusion_method == "add":
            return features_2d + features_3d

        elif self.fusion_method == "resize_and_add":
            h_2d_actual = features_2d.size(1)
            w_2d_actual = features_2d.size(2)
            h_3d_actual = features_3d.size(1)
            w_3d_actual = features_3d.size(2)

            if h_3d_actual != h_2d_actual or w_3d_actual != w_2d_actual:
                raise ValueError(
                    f"For resize_and_add, features_3d grid size ({h_3d_actual}, {w_3d_actual}) "
                    f"must match features_2d grid size ({h_2d_actual}, {w_2d_actual})."
                )
            return features_2d + features_3d

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


class GeometryFeatureMerger(nn.Module):
    """Merger for geometry features — spatial merge with MLP projection."""

    def __init__(self, output_dim: int, hidden_dim: int, context_dim: int,
                 spatial_merge_size: int = 2, merger_type: str = "mlp"):
        super().__init__()
        self.merger_type = merger_type
        self.input_dim = context_dim * (spatial_merge_size ** 2)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.merge_size = spatial_merge_size

        if merger_type == "mlp":
            self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        elif merger_type == "avg":
            self.mlp = nn.Sequential(
                nn.Linear(context_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        else:
            raise ValueError(f"Unknown merger type: {merger_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_image, h_patch, w_patch, dim = x.shape
        x = x[:, :h_patch // self.merge_size * self.merge_size, :w_patch // self.merge_size * self.merge_size, :]
        x = x.reshape(n_image, h_patch // self.merge_size, self.merge_size, w_patch // self.merge_size, self.merge_size, dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        if self.merger_type == "mlp":
            x = self.mlp(self.ln_q(x).view(-1, self.input_dim))
        elif self.merger_type == "avg":
            x = x.mean(dim=(3, 4))
            x = x.view(-1, dim)
            x = self.mlp(x)
        x = x.reshape(n_image, h_patch // self.merge_size, w_patch // self.merge_size, -1)
        return x
