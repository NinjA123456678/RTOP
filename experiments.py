from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import AblationConfig, ablation_case
from .errors import NeedsVerificationError


@dataclass(frozen=True)
class MainExperimentSpec:
    qos_type: str
    density: float
    metrics: tuple[str, str] = ("MAE", "RMSE")

    def __post_init__(self) -> None:
        if self.qos_type not in {"rt", "tp"}:
            raise ValueError("qos_type must be rt or tp")
        if not 0.0 < self.density <= 1.0:
            raise ValueError("density must be supplied explicitly in (0, 1]")


@dataclass(frozen=True)
class MultiSeedExperimentSpec:
    qos_type: str
    seeds: tuple[int, ...]
    density: float = 0.10

    def __post_init__(self) -> None:
        if self.qos_type not in {"rt", "tp"}:
            raise ValueError("qos_type must be rt or tp")
        if not self.seeds:
            raise ValueError("--seeds is required; the response does not list the five values")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")


@dataclass(frozen=True)
class RobustnessExperimentSpec:
    qos_type: str
    seed: int = 42
    density: float = 0.10
    anomaly_ratios: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
    factors: tuple[float, float] = (0.5, 1.5)


@dataclass
class AnomalyInjectionResult:
    values: np.ndarray
    anomaly_mask: np.ndarray
    applied_factors: np.ndarray
    eligible_count: int
    injected_count: int


def inject_controlled_anomalies(
    values: np.ndarray,
    observable_input_mask: np.ndarray,
    *,
    anomaly_ratio: float,
    seed: int,
    protected_target_mask: np.ndarray,
    factors: tuple[float, float] = (0.5, 1.5),
) -> AnomalyInjectionResult:
    """Inject only into model-observable inputs, never protected targets."""

    values = np.asarray(values)
    observable_input_mask = np.asarray(observable_input_mask, dtype=np.bool_)
    if values.shape != observable_input_mask.shape:
        raise ValueError("values and observable_input_mask must share a shape")
    if not 0.0 <= anomaly_ratio <= 1.0:
        raise ValueError("anomaly_ratio must be in [0, 1]")
    eligible = observable_input_mask.copy()
    protected = np.asarray(protected_target_mask, dtype=np.bool_)
    if protected.shape != values.shape:
        raise ValueError("protected_target_mask must align with values")
    eligible &= ~protected
    eligible_flat = np.flatnonzero(eligible.reshape(-1))
    injected_count = int(round(eligible_flat.size * anomaly_ratio))
    result = values.copy()
    anomaly_mask = np.zeros_like(observable_input_mask)
    applied_factors = np.ones(values.shape, dtype=np.float32)
    if injected_count:
        rng = np.random.default_rng(seed)
        selected = rng.choice(eligible_flat, size=injected_count, replace=False)
        selected_factors = rng.choice(np.asarray(factors, dtype=np.float32), size=injected_count)
        result.reshape(-1)[selected] *= selected_factors.astype(result.dtype, copy=False)
        anomaly_mask.reshape(-1)[selected] = True
        applied_factors.reshape(-1)[selected] = selected_factors
    return AnomalyInjectionResult(
        values=result,
        anomaly_mask=anomaly_mask,
        applied_factors=applied_factors,
        eligible_count=int(eligible_flat.size),
        injected_count=injected_count,
    )


def anomaly_diagnostics(
    anomaly_scores: np.ndarray,
    confidence_weights: np.ndarray,
    anomaly_mask: np.ndarray,
    observed_mask: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    arrays = [np.asarray(value) for value in (anomaly_scores, confidence_weights, anomaly_mask, observed_mask)]
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("all diagnostic arrays must share a shape")
    scores, confidence, anomalies, observed = arrays
    anomalies = anomalies.astype(bool) & observed.astype(bool)
    normal = observed.astype(bool) & ~anomalies

    def summarize(mask: np.ndarray) -> dict[str, float | int]:
        if not np.any(mask):
            return {
                "count": 0,
                "anomaly_score_mean": float("nan"),
                "anomaly_score_std": float("nan"),
                "confidence_mean": float("nan"),
                "confidence_std": float("nan"),
            }
        return {
            "count": int(mask.sum()),
            "anomaly_score_mean": float(np.mean(scores[mask])),
            "anomaly_score_std": float(np.std(scores[mask])),
            "confidence_mean": float(np.mean(confidence[mask])),
            "confidence_std": float(np.std(confidence[mask])),
        }

    return {"normal": summarize(normal), "anomalous": summarize(anomalies)}


def export_anomaly_diagnostics(
    diagnostics: dict[str, dict[str, float | int]], path: str | Path
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "count",
                "anomaly_score_mean",
                "anomaly_score_std",
                "confidence_mean",
                "confidence_std",
            ],
        )
        writer.writeheader()
        for group in ("normal", "anomalous"):
            writer.writerow({"group": group, **diagnostics[group]})


def mean_and_std(
    results: Sequence[dict[str, float]], *, ddof: int | None = None
) -> dict[str, dict[str, float]]:
    if not results:
        raise ValueError("at least one seed result is required")
    if ddof is None:
        raise NeedsVerificationError(
            "The response does not state population (ddof=0) versus sample (ddof=1) std."
        )
    if ddof not in {0, 1}:
        raise ValueError("ddof must be 0 or 1")
    keys = set(results[0])
    if any(set(result) != keys for result in results):
        raise ValueError("all seed results must contain identical metrics")
    return {
        key: {
            "mean": float(np.mean([result[key] for result in results])),
            "std": float(np.std([result[key] for result in results], ddof=ddof)),
        }
        for key in sorted(keys)
    }


def write_experiment_manifest(spec: object, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(spec) if hasattr(spec, "__dataclass_fields__") else spec
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def all_ablation_cases() -> tuple[AblationConfig, ...]:
    return tuple(ablation_case(case) for case in range(1, 9))
