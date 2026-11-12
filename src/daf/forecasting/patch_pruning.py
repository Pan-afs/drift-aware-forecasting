"""Input patch pruning for patch-based forecasters such as Chronos-Bolt.

Chronos-Bolt cuts the context into non-overlapping patches of
``patch_size`` steps, aligned to the *end* of the context (the start is
left-padded), and feeds one encoder token per patch. Encoder cost grows with
the number of tokens, quadratically in self-attention, so dropping
uninformative patches is a direct speed-up.

Pruning happens on the raw context before the model sees it: whole patches
are removed and the survivors are concatenated in chronological order. Patch
boundaries stay aligned with the model's own patching and the model weights
are untouched, so pruning combines freely with int8 quantization.

Strategies (the newest ``min_recent`` patches are always kept):

``recent``
    Keep the newest patches only. Equivalent to a shorter context; the
    baseline the other strategies must beat.
``seasonal``
    Prefer older patches that fall at the same point of the seasonal cycle as
    the forecast horizon (the same hours on previous days or weeks), i.e. the
    history a seasonal model would actually look up.
``saliency``
    Prefer older patches carrying the most information: high variability
    inside the patch, or a patch mean that breaks from its neighbours (peaks,
    level shifts). Flat stretches and straight trends are dropped first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

STRATEGIES = ("recent", "seasonal", "saliency")


@dataclass(frozen=True)
class PatchPruner:
    keep_ratio: float = 0.5
    strategy: str = "saliency"
    patch_size: int = 16
    min_recent: int = 4
    season: int = 1
    horizon: int = 64

    def __post_init__(self) -> None:
        if not 0 < self.keep_ratio <= 1:
            raise ValueError(f"keep_ratio must be in (0, 1], got {self.keep_ratio}")
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {self.strategy!r}")
        if self.patch_size < 1 or self.min_recent < 0:
            raise ValueError("patch_size must be >= 1 and min_recent >= 0")

    @classmethod
    def for_forecaster(cls, forecaster: Any, **kwargs: Any) -> PatchPruner:
        """A pruner whose patch size matches the forecaster's input patches."""
        return cls(patch_size=forecaster.patch_size, **kwargs)

    # -- patching ----------------------------------------------------------

    def split(self, context: np.ndarray) -> tuple[np.ndarray, int]:
        """(n_patches, patch_size) array, left-padded with NaN like the model."""
        x = np.asarray(context, dtype=np.float32)
        n_patches = -(-len(x) // self.patch_size)
        pad = n_patches * self.patch_size - len(x)
        padded = np.concatenate([np.full(pad, np.nan, dtype=np.float32), x])
        return padded.reshape(n_patches, self.patch_size), pad

    def n_tokens(self, length: int) -> int:
        return -(-length // self.patch_size)

    def budget(self, n_patches: int) -> int:
        k = int(np.ceil(self.keep_ratio * n_patches))
        return min(n_patches, max(self.min_recent, k))

    # -- selection ---------------------------------------------------------

    def _seasonal_scores(self, patches: np.ndarray, idx: np.ndarray) -> np.ndarray:
        n = len(patches)
        season = self.season
        if season <= 1:
            return idx.astype(float)
        # Phase (relative to the forecast origin) at which each patch starts.
        start_phase = (-(n - idx) * self.patch_size) % season
        phases = (start_phase[:, None] + np.arange(self.patch_size)) % season
        overlap = (phases < min(self.horizon, season)).sum(axis=1).astype(float)
        return overlap + 0.5 * idx / n  # ties go to the more recent patch

    @staticmethod
    def _saliency_scores(patches: np.ndarray, idx: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"), _quiet_nan_warnings():
            means = np.nanmean(patches, axis=1)
            stds = np.nanstd(patches, axis=1)
            scale = np.nanstd(patches)
        scale = scale if np.isfinite(scale) and scale > 0 else 1.0

        # Curvature of the patch-mean sequence: large at peaks and on either
        # side of a level shift, zero along a straight trend.
        jump = np.zeros(len(means))
        if len(means) >= 3:
            jump[1:-1] = np.abs(means[:-2] - 2 * means[1:-1] + means[2:]) / 2
            jump[0], jump[-1] = jump[1], jump[-2]
        score = (np.nan_to_num(stds) + np.nan_to_num(jump)) / scale
        return score[idx] + 1e-6 * idx / len(patches)

    def select(self, context: np.ndarray) -> np.ndarray:
        """Sorted indices of the patches to keep."""
        patches, _ = self.split(context)
        n = len(patches)
        k = self.budget(n)
        if k >= n:
            return np.arange(n)

        n_recent = min(self.min_recent, k)
        if self.strategy == "recent" or n_recent == k:
            return np.arange(n - k, n)

        older = np.arange(n - n_recent)
        if self.strategy == "seasonal":
            scores = self._seasonal_scores(patches, older)
        else:
            scores = self._saliency_scores(patches, older)
        empty = np.isnan(patches[older]).all(axis=1)
        scores = np.where(empty, -np.inf, scores)

        best = older[np.argsort(-scores, kind="stable")[: k - n_recent]]
        return np.sort(np.concatenate([best, np.arange(n - n_recent, n)]))

    def __call__(self, context: np.ndarray) -> np.ndarray:
        patches, pad = self.split(context)
        keep = self.select(context)
        if len(keep) == len(patches):
            return np.asarray(context, dtype=np.float32)
        out = patches[keep].reshape(-1)
        # Drop the padding again if the first patch survived; the model
        # re-pads to the same boundaries.
        return out[pad:] if keep[0] == 0 else out


class _quiet_nan_warnings:
    """Silence 'mean of empty slice' warnings for all-NaN patches."""

    def __enter__(self) -> None:
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc: object) -> None:
        self._ctx.__exit__(*exc)
