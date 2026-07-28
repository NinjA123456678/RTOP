from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NMFParameters:
    rank: int
    max_iter: int
    tol: float
    seed: int = 42
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.max_iter <= 0:
            raise ValueError("rank and max_iter must be explicitly set to positive values")
        if self.tol < 0 or self.epsilon <= 0:
            raise ValueError("tol must be nonnegative and epsilon positive")


@dataclass
class NMFResult:
    user_features: np.ndarray
    service_features: np.ndarray
    iterations: int
    loss: float
    loaded_from_cache: bool = False


@dataclass
class DualBranchNMFResult:
    original: NMFResult
    enhanced: NMFResult


def _validate_matrix(matrix: np.ndarray, observed_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    observed_mask = np.asarray(observed_mask, dtype=np.bool_)
    if matrix.ndim != 2 or matrix.shape != observed_mask.shape:
        raise ValueError("matrix and observed_mask must be aligned 2D arrays")
    if np.any(matrix[observed_mask] < 0):
        raise ValueError("NMF cannot factorize negative observed values")
    if not np.any(observed_mask):
        raise ValueError("at least one observed position is required")
    return matrix, observed_mask


def masked_nmf(
    matrix: np.ndarray,
    observed_mask: np.ndarray,
    parameters: NMFParameters,
) -> NMFResult:
    """Weighted multiplicative-update NMF over observed entries only."""

    matrix, observed_mask = _validate_matrix(matrix, observed_mask)
    users, services = matrix.shape
    if parameters.rank > min(users, services):
        raise ValueError("rank cannot exceed the smaller matrix dimension")
    rng = np.random.default_rng(parameters.seed)
    user_features = rng.random((users, parameters.rank)) + 0.01
    service_by_rank = rng.random((parameters.rank, services)) + 0.01
    mask = observed_mask.astype(np.float64)
    weighted_values = mask * matrix
    previous_loss = float("inf")
    loss = previous_loss

    for iteration in range(1, parameters.max_iter + 1):
        estimate = user_features @ service_by_rank
        numerator_h = user_features.T @ weighted_values
        denominator_h = user_features.T @ (mask * estimate) + parameters.epsilon
        service_by_rank *= numerator_h / denominator_h

        estimate = user_features @ service_by_rank
        numerator_w = weighted_values @ service_by_rank.T
        denominator_w = (mask * estimate) @ service_by_rank.T + parameters.epsilon
        user_features *= numerator_w / denominator_w

        estimate = user_features @ service_by_rank
        residual = mask * (matrix - estimate)
        loss = float(np.square(residual).sum() / max(1, observed_mask.sum()))
        if np.isfinite(previous_loss) and abs(previous_loss - loss) <= parameters.tol * max(1.0, previous_loss):
            break
        previous_loss = loss

    return NMFResult(
        user_features=user_features.astype(np.float32),
        service_features=service_by_rank.T.astype(np.float32),
        iterations=iteration,
        loss=loss,
    )


def _fingerprint(matrix: np.ndarray, observed_mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(matrix).view(np.uint8))
    digest.update(np.ascontiguousarray(observed_mask).view(np.uint8))
    return digest.hexdigest()


class NMFCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, *, qos_type: str, branch: str, time_slice: int) -> Path:
        if qos_type not in {"rt", "tp", "synthetic"}:
            raise ValueError("unsupported qos_type")
        if branch not in {"original", "enhanced"}:
            raise ValueError("branch must be original or enhanced")
        return self.cache_dir / f"{qos_type}_{branch}_t{time_slice:02d}.npz"

    def load(
        self,
        path: Path,
        *,
        parameters: NMFParameters,
        fingerprint: str,
    ) -> NMFResult | None:
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            if metadata["parameters"] != asdict(parameters) or metadata["fingerprint"] != fingerprint:
                return None
            return NMFResult(
                user_features=payload["user_features"],
                service_features=payload["service_features"],
                iterations=int(metadata["iterations"]),
                loss=float(metadata["loss"]),
                loaded_from_cache=True,
            )

    def save(
        self,
        path: Path,
        result: NMFResult,
        *,
        parameters: NMFParameters,
        fingerprint: str,
    ) -> None:
        metadata = {
            "parameters": asdict(parameters),
            "fingerprint": fingerprint,
            "iterations": result.iterations,
            "loss": result.loss,
        }
        np.savez_compressed(
            path,
            user_features=result.user_features,
            service_features=result.service_features,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )


class DualBranchNMFExtractor:
    """Precompute/cache original and enhanced matrix branches by time slice."""

    def __init__(
        self,
        parameters: NMFParameters,
        *,
        cache_dir: str | Path,
        qos_type: str,
    ) -> None:
        self.parameters = parameters
        self.cache = NMFCache(cache_dir)
        self.qos_type = qos_type

    def _extract(
        self,
        matrix: np.ndarray,
        observed_mask: np.ndarray,
        *,
        branch: str,
        time_slice: int,
        force: bool,
    ) -> NMFResult:
        matrix, observed_mask = _validate_matrix(matrix, observed_mask)
        fingerprint = _fingerprint(matrix, observed_mask)
        cache_path = self.cache.path_for(
            qos_type=self.qos_type, branch=branch, time_slice=time_slice
        )
        if not force:
            cached = self.cache.load(
                cache_path, parameters=self.parameters, fingerprint=fingerprint
            )
            if cached is not None:
                return cached
        result = masked_nmf(matrix, observed_mask, self.parameters)
        self.cache.save(
            cache_path,
            result,
            parameters=self.parameters,
            fingerprint=fingerprint,
        )
        return result

    def extract_slice(
        self,
        *,
        time_slice: int,
        original_matrix: np.ndarray,
        enhanced_matrix: np.ndarray,
        original_mask: np.ndarray,
        enhanced_mask: np.ndarray | None = None,
        force: bool = False,
    ) -> DualBranchNMFResult:
        enhanced_mask = original_mask if enhanced_mask is None else enhanced_mask
        return DualBranchNMFResult(
            original=self._extract(
                original_matrix,
                original_mask,
                branch="original",
                time_slice=time_slice,
                force=force,
            ),
            enhanced=self._extract(
                enhanced_matrix,
                enhanced_mask,
                branch="enhanced",
                time_slice=time_slice,
                force=force,
            ),
        )

    def precompute(
        self,
        original_matrices: np.ndarray,
        enhanced_matrices: np.ndarray,
        original_masks: np.ndarray,
        *,
        time_slices: list[int] | tuple[int, ...] | None = None,
        force: bool = False,
    ) -> dict[int, DualBranchNMFResult]:
        if original_matrices.shape != enhanced_matrices.shape or original_matrices.shape != original_masks.shape:
            raise ValueError("matrix sequences and masks must have identical shapes")
        selected = range(original_matrices.shape[0]) if time_slices is None else time_slices
        return {
            time: self.extract_slice(
                time_slice=time,
                original_matrix=original_matrices[time],
                enhanced_matrix=enhanced_matrices[time],
                original_mask=original_masks[time],
                force=force,
            )
            for time in selected
        }
