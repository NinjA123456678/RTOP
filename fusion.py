from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import AblationConfig


@dataclass
class EnhancedPairFeatures:
    user: torch.Tensor
    service: torch.Tensor
    attention_weights: torch.Tensor


class UserServiceFeatureEnhancer(nn.Module):
    """Enhance user/service tokens jointly instead of attending over length one.

    The legacy module converts one feature vector into a sequence of length one,
    making its attention degenerate.  This isolated implementation explicitly
    treats user and service vectors as two interacting tokens.  Exact paper
    layer widths remain configurable because they are absent from the response.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        attention_heads: int,
        convolution_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if feature_dim % attention_heads:
            raise ValueError("feature_dim must be divisible by attention_heads")
        if convolution_layers <= 0:
            raise ValueError("convolution_layers must be positive")
        self.attention = nn.MultiheadAttention(
            feature_dim, attention_heads, dropout=dropout, batch_first=True
        )
        convolutions: list[nn.Module] = []
        for _ in range(convolution_layers):
            convolutions.extend(
                [nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1), nn.ReLU()]
            )
        self.convolutions = nn.Sequential(*convolutions)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, user: torch.Tensor, service: torch.Tensor) -> EnhancedPairFeatures:
        if user.shape != service.shape or user.ndim != 2:
            raise ValueError("user and service features must share shape [batch, feature]")
        tokens = torch.stack((user, service), dim=1)
        attended, weights = self.attention(
            tokens, tokens, tokens, need_weights=True, average_attn_weights=False
        )
        convolved = self.convolutions(attended.transpose(1, 2)).transpose(1, 2)
        enhanced = self.norm(tokens + convolved)
        return EnhancedPairFeatures(
            user=enhanced[:, 0], service=enhanced[:, 1], attention_weights=weights
        )


@dataclass
class FusionOutput:
    feature: torch.Tensor
    active_sources: tuple[str, ...]


class MultiSourceFeatureFusion(nn.Module):
    """Fuse temporal plus original/enhanced user-service sources.

    Disabled sources are represented by zeros so all eight ablations share one
    predictor architecture rather than copying model code.
    """

    SOURCE_NAMES = (
        "temporal",
        "original_user",
        "original_service",
        "enhanced_user",
        "enhanced_service",
    )

    def __init__(self, temporal_dim: int, nmf_dim: int, *, projection_dim: int) -> None:
        super().__init__()
        self.projection_dim = projection_dim
        self.projections = nn.ModuleDict(
            {
                "temporal": nn.Linear(temporal_dim, projection_dim),
                "original_user": nn.Linear(nmf_dim, projection_dim),
                "original_service": nn.Linear(nmf_dim, projection_dim),
                "enhanced_user": nn.Linear(nmf_dim, projection_dim),
                "enhanced_service": nn.Linear(nmf_dim, projection_dim),
            }
        )
        self.norm = nn.LayerNorm(projection_dim * len(self.SOURCE_NAMES))

    @property
    def output_dim(self) -> int:
        return self.projection_dim * len(self.SOURCE_NAMES)

    def forward(
        self,
        *,
        temporal: torch.Tensor,
        original_user: torch.Tensor,
        original_service: torch.Tensor,
        enhanced_user: torch.Tensor,
        enhanced_service: torch.Tensor,
        ablation: AblationConfig,
    ) -> FusionOutput:
        batch = temporal.shape[0]
        inputs = {
            "temporal": temporal,
            "original_user": original_user,
            "original_service": original_service,
            "enhanced_user": enhanced_user,
            "enhanced_service": enhanced_service,
        }
        enabled = {
            "temporal": True,
            "original_user": ablation.use_user_service_features
            and ablation.use_original_matrix_branch,
            "original_service": ablation.use_user_service_features
            and ablation.use_original_matrix_branch,
            "enhanced_user": ablation.use_user_service_features
            and ablation.use_enhanced_matrix_branch,
            "enhanced_service": ablation.use_user_service_features
            and ablation.use_enhanced_matrix_branch,
        }
        projected: list[torch.Tensor] = []
        active: list[str] = []
        for name in self.SOURCE_NAMES:
            value = inputs[name]
            if value.shape[0] != batch:
                raise ValueError("all fusion sources must share a batch dimension")
            if enabled[name]:
                projected.append(torch.relu(self.projections[name](value)))
                active.append(name)
            else:
                projected.append(
                    torch.zeros(
                        batch,
                        self.projection_dim,
                        dtype=temporal.dtype,
                        device=temporal.device,
                    )
                )
        return FusionOutput(feature=self.norm(torch.cat(projected, dim=-1)), active_sources=tuple(active))


class PredictionFFN(nn.Module):
    def __init__(self, input_dim: int, *, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.network(feature).squeeze(-1)
