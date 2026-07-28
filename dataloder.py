from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


TRAIN_TARGET_TIMES = tuple(range(7, 47))
VALIDATION_TARGET_TIMES = tuple(range(47, 58))
TEST_TARGET_TIMES = tuple(range(58, 64))
TARGET_TIME_SPLITS: Mapping[str, tuple[int, ...]] = {
    "train": TRAIN_TARGET_TIMES,
    "validation": VALIDATION_TARGET_TIMES,
    "test": TEST_TARGET_TIMES,
}


@dataclass(frozen=True)
class QoSData:
    values: np.ndarray
    observed_mask: np.ndarray
    index_base: int
    source: Path

    def __post_init__(self) -> None:
        if self.values.shape != self.observed_mask.shape:
            raise ValueError("values and observed_mask must have identical shapes")
        if self.values.ndim != 3:
            raise ValueError("QoS data must have shape [time, user, service]")
        if self.observed_mask.dtype != np.bool_:
            raise TypeError("observed_mask must be boolean")


def _records(path: Path) -> Iterator[tuple[int, int, int, float]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 4:
                raise ValueError(f"{path}:{line_number}: expected four columns")
            try:
                user, service, time = map(int, fields[:3])
                value = float(fields[3])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid numeric record") from exc
            yield user, service, time, value


def detect_index_base(path: str | Path) -> int:
    """Detect 0-based data safely; a seen zero is decisive.

    If no zero occurs anywhere, the minimum IDs must all be one for a 1-based
    file. Mixed or otherwise ambiguous indexing is rejected.
    """

    source = Path(path)
    minima = [None, None, None]
    for user, service, time, _ in _records(source):
        ids = (user, service, time)
        if any(value < 0 for value in ids):
            raise ValueError("negative raw IDs are not supported")
        if 0 in ids:
            return 0
        minima = [value if old is None else min(old, value) for old, value in zip(minima, ids)]
    if minima == [1, 1, 1]:
        return 1
    if minima == [None, None, None]:
        raise ValueError(f"no records found in {source}")
    raise ValueError(f"cannot safely determine index base from minima {minima}")


def load_qos_file(
    path: str | Path,
    shape: tuple[int, int, int],
    *,
    index_base: int | str = "auto",
    missing_value: float = 0.0,
    dtype: np.dtype = np.float32,
) -> QoSData:
    """Load ``user service time value`` records without legacy ``-1`` wraparound."""

    source = Path(path)
    if index_base == "auto":
        base = detect_index_base(source)
    elif index_base in (0, 1):
        base = int(index_base)
    else:
        raise ValueError("index_base must be 0, 1, or 'auto'")

    num_times, num_users, num_services = shape
    values = np.full(shape, missing_value, dtype=dtype)
    present = np.zeros(shape, dtype=np.bool_)
    for raw_user, raw_service, raw_time, value in _records(source):
        user, service, time = raw_user - base, raw_service - base, raw_time - base
        if not (0 <= time < num_times and 0 <= user < num_users and 0 <= service < num_services):
            raise IndexError(
                f"raw index {(raw_user, raw_service, raw_time)} maps outside {shape}"
            )
        if present[time, user, service]:
            raise ValueError(f"duplicate record at {(raw_user, raw_service, raw_time)}")
        values[time, user, service] = value
        present[time, user, service] = True

    # The response letter defines zero entries as missing observations.  A
    # separate presence mask is retained during loading so malformed or sparse
    # test files are not confused with enumerated zero-valued records.
    observed = present & ~np.isclose(values, missing_value)
    return QoSData(values=values, observed_mask=observed, index_base=base, source=source)


def load_qos_type(
    qos_type: str,
    *,
    rt_path: str | Path = "data/rtdata.txt",
    tp_path: str | Path = "data/tpdata.txt",
    shape: tuple[int, int, int] = (64, 142, 4500),
) -> QoSData:
    if qos_type not in {"rt", "tp"}:
        raise ValueError("qos_type must be 'rt' or 'tp'")
    return load_qos_file(rt_path if qos_type == "rt" else tp_path, shape, index_base="auto")


def sample_observed_mask(
    observed_mask: np.ndarray,
    density: float,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Sample an explicit full-matrix density independently in each time slice."""

    if observed_mask.ndim != 3 or observed_mask.dtype != np.bool_:
        raise ValueError("observed_mask must be a boolean [time, user, service] array")
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    _, num_users, num_services = observed_mask.shape
    target_per_slice = int(num_users * num_services * density)
    if target_per_slice < 1:
        raise ValueError("density is too small to sample one entry per time slice")
    sampled = np.zeros_like(observed_mask)
    seed_sequence = np.random.SeedSequence(seed)
    for time, child_seed in enumerate(seed_sequence.spawn(observed_mask.shape[0])):
        candidates = np.flatnonzero(observed_mask[time].reshape(-1))
        if candidates.size < target_per_slice:
            raise ValueError(
                f"time slice {time} has {candidates.size} observed entries, fewer than "
                f"the requested full-matrix density count {target_per_slice}"
            )
        rng = np.random.default_rng(child_seed)
        chosen = rng.choice(candidates, size=target_per_slice, replace=False)
        sampled[time].reshape(-1)[chosen] = True
    return sampled


def target_times_for_split(split: str, *, history_window: int = 7) -> tuple[int, ...]:
    if history_window != 7:
        raise ValueError("supplied response confirms only F=7 for the published protocol")
    try:
        return TARGET_TIME_SPLITS[split]
    except KeyError as exc:
        raise ValueError("split must be train, validation, or test") from exc


class ChronologicalWindowDataset(Dataset):
    """Entry-level samples with strictly historical, sampled inputs."""

    def __init__(
        self,
        values: np.ndarray,
        observed_mask: np.ndarray,
        sampled_mask: np.ndarray,
        *,
        split: str,
        history_window: int = 7,
        target_times: Sequence[int] | None = None,
    ) -> None:
        if values.shape != observed_mask.shape or values.shape != sampled_mask.shape:
            raise ValueError("values, observed_mask, and sampled_mask must share a shape")
        if observed_mask.dtype != np.bool_ or sampled_mask.dtype != np.bool_:
            raise TypeError("masks must be boolean")
        if np.any(sampled_mask & ~observed_mask):
            raise ValueError("sampled_mask must be a subset of observed_mask")
        self.values = values
        self.observed_mask = observed_mask
        self.sampled_mask = sampled_mask
        self.history_window = history_window
        self.split = split
        self.target_times = tuple(
            target_times if target_times is not None else target_times_for_split(split, history_window=history_window)
        )
        if not self.target_times:
            raise ValueError("target_times cannot be empty")
        if min(self.target_times) < history_window or max(self.target_times) >= values.shape[0]:
            raise ValueError("target times are incompatible with data shape/history window")

        pair_indices: list[np.ndarray] = []
        time_indices: list[np.ndarray] = []
        for time in self.target_times:
            pairs = np.flatnonzero(sampled_mask[time].reshape(-1))
            pair_indices.append(pairs)
            time_indices.append(np.full(pairs.shape, time, dtype=np.int16))
        self._pair_indices = np.concatenate(pair_indices) if pair_indices else np.empty(0, dtype=np.int64)
        self._time_indices = np.concatenate(time_indices) if time_indices else np.empty(0, dtype=np.int16)

    def __len__(self) -> int:
        return int(self._pair_indices.size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        target_time = int(self._time_indices[index])
        flat_pair = int(self._pair_indices[index])
        num_services = self.values.shape[2]
        user, service = divmod(flat_pair, num_services)
        start = target_time - self.history_window
        input_times = np.arange(start, target_time, dtype=np.int64)
        history_mask = self.sampled_mask[start:target_time, user, service]
        history = self.values[start:target_time, user, service]
        visible_history = np.where(history_mask, history, 0.0)
        return {
            "input_values": torch.as_tensor(visible_history, dtype=torch.float32),
            "input_observed_mask": torch.as_tensor(history_mask, dtype=torch.bool),
            "input_times": torch.as_tensor(input_times, dtype=torch.long),
            "target_value": torch.tensor(self.values[target_time, user, service], dtype=torch.float32),
            "target_time": torch.tensor(target_time, dtype=torch.long),
            "user": torch.tensor(user, dtype=torch.long),
            "service": torch.tensor(service, dtype=torch.long),
        }


def build_chronological_datasets(
    data: QoSData,
    sampled_mask: np.ndarray,
    *,
    history_window: int = 7,
) -> dict[str, ChronologicalWindowDataset]:
    return {
        split: ChronologicalWindowDataset(
            data.values,
            data.observed_mask,
            sampled_mask,
            split=split,
            history_window=history_window,
        )
        for split in ("train", "validation", "test")
    }
