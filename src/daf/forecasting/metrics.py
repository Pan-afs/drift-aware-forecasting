"""Point and probabilistic forecast metrics.

MASE and WQL follow the definitions used in the Chronos paper (Ansari et
al., 2024) so numbers are comparable with published results:

* MASE: mean absolute error of the median forecast, scaled by the in-sample
  mean absolute error of the seasonal-naive forecast on the context.
* WQL:  weighted quantile loss, the pinball loss summed over the horizon,
  scaled by the sum of absolute target values and averaged over quantiles.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def seasonal_error(context: np.ndarray, season: int = 1) -> float:
    """Mean absolute seasonal-naive error on the context (MASE denominator)."""
    x = np.asarray(context, dtype=float)
    if season >= len(x):
        season = 1
    diffs = np.abs(x[season:] - x[:-season])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return np.nan
    scale = float(diffs.mean())
    return scale if scale > 0 else np.nan


def mase(y_true: np.ndarray, y_pred: np.ndarray, context: np.ndarray, season: int = 1) -> float:
    y_true = np.asarray(y_true, dtype=float)
    err = np.abs(y_true - np.asarray(y_pred, dtype=float))
    err = err[np.isfinite(err)]
    if err.size == 0:
        return np.nan
    return float(err.mean() / seasonal_error(context, season))


def pinball_loss(y_true: np.ndarray, y_quantile: np.ndarray, q: float) -> np.ndarray:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_quantile, dtype=float)
    return np.maximum(q * diff, (q - 1) * diff)


def wql(y_true: np.ndarray, q_pred: np.ndarray, quantiles: Sequence[float]) -> float:
    """Weighted quantile loss for one series; ``q_pred`` has shape (Q, H)."""
    y_true = np.asarray(y_true, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)
    if q_pred.shape != (len(quantiles), len(y_true)):
        raise ValueError(f"q_pred shape {q_pred.shape} != ({len(quantiles)}, {len(y_true)})")
    mask = np.isfinite(y_true)
    denom = np.abs(y_true[mask]).sum()
    if not mask.any() or denom == 0:
        return np.nan
    losses = [
        2 * pinball_loss(y_true[mask], q_pred[i, mask], q).sum() for i, q in enumerate(quantiles)
    ]
    return float(np.mean(losses) / denom)


def median_index(quantiles: Sequence[float]) -> int:
    try:
        return list(quantiles).index(0.5)
    except ValueError:
        raise ValueError("quantile levels must include 0.5 for MASE") from None


def gmean(values: Sequence[float]) -> float:
    """Geometric mean of positive finite values (used for relative scores)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    return float(np.exp(np.log(arr).mean())) if arr.size else np.nan
