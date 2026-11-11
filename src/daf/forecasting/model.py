"""Forecasters with a common quantile interface.

Every forecaster maps a batch of 1-D contexts (possibly of different lengths,
possibly containing NaN) to quantile forecasts of shape (batch, Q, horizon).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

from daf.forecasting.metrics import DEFAULT_QUANTILES

ContextTransform = Callable[[np.ndarray], np.ndarray]


class Forecaster(Protocol):
    name: str

    def predict_quantiles(
        self,
        contexts: Sequence[np.ndarray],
        prediction_length: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> np.ndarray: ...


class SeasonalNaive:
    """Repeat the last season; quantiles from the context's own seasonal errors.

    A cheap, strong baseline for series with stable cycles (mobile traffic)
    and the reference point MASE is built on.
    """

    def __init__(self, season: int = 1):
        self.season = max(int(season), 1)
        self.name = f"seasonal_naive(m={self.season})"

    def _one(self, context: np.ndarray, h: int, quantiles: Sequence[float]) -> np.ndarray:
        x = np.asarray(context, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.full((len(quantiles), h), np.nan)
        m = self.season if x.size > self.season else 1
        last = x[-m:]
        point = np.resize(last, h)
        errors = x[m:] - x[:-m]
        offsets = np.quantile(errors, quantiles) if errors.size else np.zeros(len(quantiles))
        return point[None, :] + offsets[:, None]

    def predict_quantiles(self, contexts, prediction_length, quantiles=DEFAULT_QUANTILES):
        return np.stack([self._one(c, prediction_length, quantiles) for c in contexts])


class ChronosBolt:
    """Thin wrapper around the ``chronos-forecasting`` Bolt pipeline (CPU).

    ``context_transform`` is applied to every context before inference; the
    patch pruner in :mod:`daf.forecasting.patch_pruning` plugs in here.
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-bolt-small",
        pipeline: Any = None,
        name: str | None = None,
        context_transform: ContextTransform | None = None,
        device: str = "cpu",
    ):
        if pipeline is None:
            import torch
            from chronos import BaseChronosPipeline

            pipeline = BaseChronosPipeline.from_pretrained(
                model_id, device_map=device, torch_dtype=torch.float32
            )
        self.model_id = model_id
        self.pipeline = pipeline
        self.name = name or model_id.rsplit("/", 1)[-1]
        self.context_transform = context_transform

    @property
    def model(self) -> Any:
        return self.pipeline.model

    @property
    def patch_size(self) -> int:
        config = getattr(self.model, "chronos_config", None)
        return int(getattr(config, "input_patch_size", 16))

    @property
    def max_context(self) -> int:
        config = getattr(self.model, "chronos_config", None)
        return int(getattr(config, "context_length", 2048))

    def prepare(self, contexts: Sequence[np.ndarray]) -> list[np.ndarray]:
        contexts = [np.asarray(c, dtype=np.float32)[-self.max_context :] for c in contexts]
        if self.context_transform is not None:
            contexts = [self.context_transform(c) for c in contexts]
        return contexts

    def predict_quantiles(self, contexts, prediction_length, quantiles=DEFAULT_QUANTILES):
        import torch

        tensors = [torch.from_numpy(np.ascontiguousarray(c)) for c in self.prepare(contexts)]
        with torch.inference_mode():
            q, _mean = self.pipeline.predict_quantiles(
                tensors, prediction_length=prediction_length, quantile_levels=list(quantiles)
            )
        # (batch, horizon, Q) -> (batch, Q, horizon)
        return q.permute(0, 2, 1).float().cpu().numpy()
