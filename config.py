from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .errors import NeedsVerificationError


@dataclass(frozen=True)
class DataConfig:
    """Configuration for the isolated chronological data pipeline."""

    qos_type: str
    density: Optional[float] = None
    rt_path: Path = Path("data/rtdata.txt")
    tp_path: Path = Path("data/tpdata.txt")
    num_users: int = 142
    num_services: int = 4500
    num_times: int = 64
    history_window: int = 7
    seed: int = 42

    def __post_init__(self) -> None:
        if self.qos_type not in {"rt", "tp"}:
            raise ValueError("qos_type must be 'rt' or 'tp'")
        if self.density is not None and not 0.0 < self.density <= 1.0:
            raise ValueError("density must be in (0, 1]")
        if self.history_window <= 0:
            raise ValueError("history_window must be positive")

    @property
    def data_path(self) -> Path:
        return self.rt_path if self.qos_type == "rt" else self.tp_path

    def require_density(self) -> float:
        if self.density is None:
            raise NeedsVerificationError(
                "Matrix density has no paper-confirmed default; pass it explicitly."
            )
        return self.density


@dataclass(frozen=True)
class ModelConfig:
    """Response-confirmed defaults plus explicitly unresolved NMF settings."""

    history_window: int = 7
    masking_probability: float = 0.1
    stacked_autoencoders: int = 2
    conv_kernel_sizes: tuple[int, int] = (3, 5)
    conv_channels_per_branch: int = 16
    attention_heads: int = 4
    ffn_hidden_dim: int = 128
    dropout: float = 0.2
    reconstruction_loss_weight: float = 0.1
    nmf_rank: Optional[int] = None
    nmf_max_iter: Optional[int] = None
    confidence_normalization: str = "NEEDS_VERIFICATION"
    confidence_attention_rule: str = "NEEDS_VERIFICATION"

    def require_nmf(self) -> tuple[int, int]:
        if self.nmf_rank is None or self.nmf_max_iter is None:
            raise NeedsVerificationError(
                "NMF rank and iteration count are not stated in the supplied response; "
                "pass both explicitly."
            )
        return self.nmf_rank, self.nmf_max_iter


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 100
    seed: int = 42
    validate_every_epoch: bool = True
    early_stopping: bool = False
    selection_metric: Optional[str] = None


@dataclass(frozen=True)
class AblationConfig:
    case: int
    use_sdae: bool = True
    use_temporal_attention: bool = True
    use_user_service_features: bool = True
    use_reconstruction_fusion: bool = True
    use_confidence_attention: bool = True
    use_confidence_prediction_loss: bool = True
    use_original_matrix_branch: bool = True
    use_enhanced_matrix_branch: bool = True


def ablation_case(case: int) -> AblationConfig:
    """Return the single-source-of-truth feature flags for Cases 1--8."""

    if case not in range(1, 9):
        raise ValueError("ablation case must be an integer from 1 to 8")
    flags = dict(
        case=case,
        use_sdae=True,
        use_temporal_attention=True,
        use_user_service_features=True,
        use_reconstruction_fusion=True,
        use_confidence_attention=True,
        use_confidence_prediction_loss=True,
        use_original_matrix_branch=True,
        use_enhanced_matrix_branch=True,
    )
    if case == 1:
        flags.update(
            use_sdae=False,
            use_reconstruction_fusion=False,
            use_confidence_attention=False,
            use_confidence_prediction_loss=False,
            use_enhanced_matrix_branch=False,
        )
    elif case == 2:
        flags["use_temporal_attention"] = False
        flags["use_confidence_attention"] = False
    elif case == 3:
        flags.update(
            use_user_service_features=False,
            use_original_matrix_branch=False,
            use_enhanced_matrix_branch=False,
        )
    elif case == 4:
        flags["use_reconstruction_fusion"] = False
    elif case == 5:
        flags["use_confidence_attention"] = False
    elif case == 6:
        flags["use_confidence_prediction_loss"] = False
    elif case == 7:
        flags["use_enhanced_matrix_branch"] = False
    return AblationConfig(**flags)
