"""Turn a raw series into standardised surprises and locate sudden changes.

1. Seasonal adjustment: subtract a robust per-phase seasonal profile.
2. Causal baseline: the value expected from the previous ``window`` points,
   a rolling median level extrapolated with a robust local slope.
3. Surprise: (value - baseline) divided by a trailing robust scale (MAD).
4. Candidate events: points whose |z| exceeds a threshold, plus the onset of
   any run flagged by a two-sided CUSUM. The CUSUM catches moderate but
   persistent shifts that never produce a single large z.

Classifying an event as a data error or a real shift happens in
:mod:`daf.quality.classify`; this module only finds *where* something changed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class DetectorConfig:
    z_threshold: float = 4.5
    cusum_k: float = 0.5
    cusum_h: float = 10.0
    # Points used for the causal baseline; default max(period, 24).
    window: int | None = None
    # Points after an event used to judge whether it persists; default
    # clip(window // 4, 8, 48).
    persistence_window: int | None = None
    # Share of post-event points that must stay at the new level for drift.
    persistence_min: float = 0.6
    # |log10(new / old)| this close to a non-zero integer => unit/scale error.
    unit_tolerance: float = 0.08
    # Peer events within this many steps of the target event count as support.
    peer_tolerance: int = 3
    peer_min_support: float = 0.3

    def baseline_window(self, period: int) -> int:
        return self.window or max(period, 24)

    def post_window(self, period: int) -> int:
        if self.persistence_window:
            return self.persistence_window
        return int(np.clip(self.baseline_window(period) // 4, 8, 48))


@dataclass
class Decomposition:
    index: pd.DatetimeIndex
    values: np.ndarray  # raw
    adjusted: np.ndarray  # seasonally adjusted
    baseline: np.ndarray  # causal expected value of `adjusted`
    slope: np.ndarray  # local slope per step
    scale: np.ndarray  # trailing robust sigma of residuals
    z: np.ndarray  # (adjusted - baseline) / scale


@dataclass(frozen=True)
class ChangeEvent:
    start: int
    end: int
    peak: int
    peak_z: float

    @property
    def direction(self) -> int:
        return 1 if self.peak_z > 0 else -1


def seasonal_profile(values: np.ndarray, period: int) -> np.ndarray:
    """Per-phase median of the detrended series, aligned to ``values``."""
    n = len(values)
    if period <= 1 or n < 2 * period:
        return np.zeros(n)
    s = pd.Series(values)
    trend = s.rolling(period, center=True, min_periods=max(period // 2, 1)).median()
    phase = np.arange(n) % period
    profile = (s - trend).groupby(phase).median()
    profile = profile - profile.mean()
    return profile.reindex(phase).fillna(0.0).to_numpy()


def _rolling_mad(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=max(window // 2, 3)).apply(
        lambda w: np.nanmedian(np.abs(w - np.nanmedian(w))), raw=True
    )


def decompose(s: pd.Series, period: int = 1, config: DetectorConfig | None = None) -> Decomposition:
    config = config or DetectorConfig()
    window = config.baseline_window(period)
    values = s.to_numpy(dtype=float)
    adjusted = values - seasonal_profile(values, period)

    # Level = median of the most recent half window; slope = difference between
    # the medians of the two halves. Both are robust to isolated outliers, and
    # unlike a median of one-step differences the slope noise is not blown up
    # when extrapolating forward.
    adj = pd.Series(adjusted)
    half = max(window // 2, 2)
    recent = adj.rolling(half, min_periods=max(half // 2, 2)).median().shift(1)
    older = recent.shift(half)
    slope = ((recent - older) / half).fillna(0.0)
    # `recent` is centred (half + 1) / 2 steps behind t.
    baseline = recent + slope * (half + 1) / 2
    resid = adj - baseline

    global_sigma = MAD_TO_SIGMA * np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    if not np.isfinite(global_sigma):
        global_sigma = 0.0
    floor = max(0.1 * global_sigma, 1e-9 * (1.0 + np.nanmedian(np.abs(adjusted))))
    scale = (MAD_TO_SIGMA * _rolling_mad(resid, 4 * window).shift(1)).fillna(global_sigma)
    scale = np.maximum(scale.to_numpy(), floor)

    return Decomposition(
        index=s.index,
        values=values,
        adjusted=adjusted,
        baseline=baseline.to_numpy(),
        slope=slope.to_numpy(),
        scale=scale,
        z=(resid / scale).to_numpy(),
    )


def cusum_onsets(z: np.ndarray, k: float = 0.5, h: float = 10.0) -> list[int]:
    """Two-sided tabular CUSUM on standardised residuals.

    Returns the estimated onset (last index where the statistic was zero) of
    each alarm. Inputs are clipped at +-h so one huge outlier cannot leave
    residual mass in the statistic.
    """
    onsets: list[int] = []
    pos = neg = 0.0
    pos_start = neg_start = 0
    for i, v in enumerate(z):
        if not np.isfinite(v):
            continue
        v = float(np.clip(v, -h, h))
        new_pos = max(0.0, pos + v - k)
        new_neg = max(0.0, neg - v - k)
        if pos == 0.0 and new_pos > 0.0:
            pos_start = i
        if neg == 0.0 and new_neg > 0.0:
            neg_start = i
        pos, neg = new_pos, new_neg
        if pos > h or neg > h:
            onsets.append(pos_start if pos > h else neg_start)
            pos = neg = 0.0
    return onsets


def detect_events(dec: Decomposition, config: DetectorConfig, period: int = 1) -> list[ChangeEvent]:
    z = dec.z
    finite = np.isfinite(z)
    big = np.flatnonzero(finite & (np.abs(np.where(finite, z, 0.0)) > config.z_threshold))
    candidates = sorted(set(big.tolist()) | set(cusum_onsets(z, config.cusum_k, config.cusum_h)))
    if not candidates:
        return []

    merge_gap = max(3, config.post_window(period) // 2)
    groups: list[list[int]] = [[candidates[0]]]
    for idx in candidates[1:]:
        if idx - groups[-1][-1] <= merge_gap:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    events = []
    for group in groups:
        first, end = group[0], group[-1]
        seg = z[first : end + 1]
        abs_seg = np.where(np.isfinite(seg), np.abs(seg), -np.inf)
        peak = first + int(np.argmax(abs_seg))
        peak_z = float(z[peak])
        # A CUSUM onset can sit a few noisy steps before the real change, so
        # the event starts at the first point clearly on the peak's side.
        bar = min(config.z_threshold / 2, abs(peak_z) / 2)
        strong = np.isfinite(seg) & (np.sign(seg) == np.sign(peak_z)) & (np.abs(seg) >= bar)
        start = first + int(np.argmax(strong)) if strong.any() else peak
        events.append(ChangeEvent(start=start, end=end, peak=peak, peak_z=peak_z))
    return events
