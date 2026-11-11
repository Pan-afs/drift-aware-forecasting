"""Rolling-origin evaluation windows.

The last ``n_windows * stride`` points of every series are held out; each
window forecasts ``prediction_length`` steps from its origin using only the
data before it. Windows never overlap the future of an earlier origin's
context, so there is no leakage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvalWindow:
    series_id: str
    origin: pd.Timestamp
    context: np.ndarray
    target: np.ndarray


def rolling_windows(
    series: Mapping[str, pd.Series],
    prediction_length: int,
    n_windows: int = 1,
    context_length: int | None = None,
    stride: int | None = None,
    min_context: int | None = None,
) -> list[EvalWindow]:
    stride = stride or prediction_length
    min_context = min_context or 2 * prediction_length
    windows = []
    for sid, s in series.items():
        s = s.sort_index()
        values = s.to_numpy(dtype=float)
        n = len(values)
        for w in range(n_windows):
            end = n - (n_windows - 1 - w) * stride
            origin = end - prediction_length
            if origin < min_context:
                continue
            start = 0 if context_length is None else max(0, origin - context_length)
            target = values[origin:end]
            if not np.isfinite(target).any():
                continue
            windows.append(EvalWindow(sid, s.index[origin], values[start:origin], target))
    return windows
