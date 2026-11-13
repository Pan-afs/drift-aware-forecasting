"""Accuracy vs. speed benchmark for compressed Chronos-Bolt variants.

Every variant forecasts the same rolling-origin windows. Accuracy is MASE
(median forecast) and WQL (all quantiles); speed is per-batch latency and
throughput on CPU; size is the serialized state dict. A seasonal-naive row is
included as the floor any model has to beat.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from daf.forecasting.metrics import DEFAULT_QUANTILES, mase, median_index, wql
from daf.forecasting.model import ChronosBolt, Forecaster, SeasonalNaive
from daf.forecasting.patch_pruning import PatchPruner
from daf.forecasting.windows import EvalWindow, rolling_windows


@dataclass(frozen=True)
class Variant:
    name: str
    quantize: bool = False
    strategy: str | None = None
    keep_ratio: float = 1.0


DEFAULT_VARIANTS = (
    Variant("fp32"),
    Variant("int8", quantize=True),
    Variant("fp32+recent50", strategy="recent", keep_ratio=0.5),
    Variant("fp32+seasonal50", strategy="seasonal", keep_ratio=0.5),
    Variant("fp32+saliency50", strategy="saliency", keep_ratio=0.5),
    Variant("int8+seasonal50", quantize=True, strategy="seasonal", keep_ratio=0.5),
    Variant("int8+saliency50", quantize=True, strategy="saliency", keep_ratio=0.5),
    Variant("int8+saliency25", quantize=True, strategy="saliency", keep_ratio=0.25),
)


@dataclass(frozen=True)
class BenchmarkConfig:
    model_id: str = "amazon/chronos-bolt-small"
    prediction_length: int = 24
    context_length: int = 512
    n_windows: int = 4
    batch_size: int = 32
    season: int = 1
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    warmup: int = 1
    repeats: int = 3
    num_threads: int | None = None
    min_recent: int = 4


def evaluate(
    forecaster: Forecaster,
    windows: Sequence[EvalWindow],
    config: BenchmarkConfig,
    seasons: Mapping[str, int] | None = None,
) -> dict[str, float]:
    """Time ``forecaster`` on all windows and score its forecasts."""
    h, q = config.prediction_length, config.quantiles
    contexts = [w.context for w in windows]
    size = config.batch_size
    batches = [contexts[i : i + size] for i in range(0, len(contexts), size)]

    for _ in range(config.warmup):
        forecaster.predict_quantiles(batches[0], h, q)

    timings: list[float] = []
    preds = np.empty(0)
    for _ in range(max(config.repeats, 1)):
        outputs = []
        for batch in batches:
            start = time.perf_counter()
            outputs.append(forecaster.predict_quantiles(batch, h, q))
            timings.append(time.perf_counter() - start)
        preds = np.concatenate(outputs)

    mid = median_index(q)
    seasons = seasons or {}
    mases = [
        mase(w.target, p[mid], w.context, seasons.get(w.series_id, config.season))
        for w, p in zip(windows, preds)
    ]
    wqls = [wql(w.target, p, q) for w, p in zip(windows, preds)]
    batch_ms = np.asarray(timings) * 1000
    seconds_per_pass = sum(timings) / max(config.repeats, 1)
    return {
        "mase": float(np.nanmean(mases)),
        "wql": float(np.nanmean(wqls)),
        "latency_ms_p50": float(np.percentile(batch_ms, 50)),
        "latency_ms_p95": float(np.percentile(batch_ms, 95)),
        "throughput": len(windows) / seconds_per_pass if seconds_per_pass > 0 else np.nan,
    }


def add_relative_columns(results: pd.DataFrame, reference: str = "fp32") -> pd.DataFrame:
    """Accuracy and speed of each row relative to the reference variant."""
    out = results.copy()
    ref = out.loc[out["variant"] == reference]
    if ref.empty:
        return out
    ref = ref.iloc[0]
    out["mase_vs_ref"] = out["mase"] / ref["mase"]
    out["wql_vs_ref"] = out["wql"] / ref["wql"]
    out["speedup"] = ref["latency_ms_p50"] / out["latency_ms_p50"]
    return out


def run_benchmark(
    series: Mapping[str, pd.Series],
    config: BenchmarkConfig,
    variants: Sequence[Variant] = DEFAULT_VARIANTS,
    seasons: Mapping[str, int] | None = None,
    base: ChronosBolt | None = None,
) -> pd.DataFrame:
    import torch

    from daf.forecasting.quantization import model_size_mb, quantize_forecaster

    if config.num_threads:
        torch.set_num_threads(config.num_threads)
    torch.manual_seed(0)

    windows = rolling_windows(
        series, config.prediction_length, config.n_windows, config.context_length
    )
    if not windows:
        raise ValueError("no evaluation windows: series too short for this configuration")

    rows = [
        {
            "variant": "seasonal_naive",
            "quantized": False,
            "strategy": None,
            "keep_ratio": 1.0,
            "model_size_mb": 0.0,
            "encoder_tokens": np.nan,
            **evaluate(SeasonalNaive(config.season), windows, config, seasons),
        }
    ]

    base = base or ChronosBolt(config.model_id)
    models = {False: base}
    sizes = {False: model_size_mb(base.model)}
    for variant in variants:
        if variant.quantize not in models:
            models[True] = quantize_forecaster(base)
            sizes[True] = model_size_mb(models[True].model)
        source = models[variant.quantize]

        pruner = None
        if variant.strategy:
            pruner = PatchPruner.for_forecaster(
                source,
                keep_ratio=variant.keep_ratio,
                strategy=variant.strategy,
                min_recent=config.min_recent,
                season=config.season,
                horizon=config.prediction_length,
            )
        forecaster = ChronosBolt(
            model_id=config.model_id,
            pipeline=source.pipeline,
            name=variant.name,
            context_transform=pruner,
        )
        prepared = forecaster.prepare([w.context for w in windows])
        tokens = np.mean([-(-len(c) // forecaster.patch_size) for c in prepared])

        rows.append(
            {
                "variant": variant.name,
                "quantized": variant.quantize,
                "strategy": variant.strategy,
                "keep_ratio": variant.keep_ratio,
                "model_size_mb": sizes[variant.quantize],
                "encoder_tokens": float(tokens),
                **evaluate(forecaster, windows, config, seasons),
            }
        )

    results = add_relative_columns(pd.DataFrame(rows))
    results.attrs["config"] = asdict(config)
    results.attrs["n_windows"] = len(windows)
    results.attrs["n_series"] = len({w.series_id for w in windows})
    return results


def to_markdown(results: pd.DataFrame, title: str = "Benchmark") -> str:
    """A compact Markdown report (no extra dependencies)."""
    columns = [
        ("variant", "Variant", "{}"),
        ("mase", "MASE", "{:.3f}"),
        ("wql", "WQL", "{:.3f}"),
        ("latency_ms_p50", "p50 ms/batch", "{:.1f}"),
        ("speedup", "Speed-up", "{:.2f}x"),
        ("encoder_tokens", "Tokens", "{:.0f}"),
        ("model_size_mb", "Size MB", "{:.1f}"),
    ]
    columns = [c for c in columns if c[0] in results.columns]

    def fmt(value: object, pattern: str) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "-"
        return pattern.format(value)

    lines = [f"## {title}", ""]
    cfg = results.attrs.get("config")
    if cfg:
        lines += [
            f"{cfg['model_id']}, context {cfg['context_length']}, horizon "
            f"{cfg['prediction_length']}, {results.attrs.get('n_series')} series x "
            f"{cfg['n_windows']} windows, batch {cfg['batch_size']}.",
            "",
        ]
    lines.append("| " + " | ".join(c[1] for c in columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for _, row in results.iterrows():
        lines.append("| " + " | ".join(fmt(row[c[0]], c[2]) for c in columns) + " |")
    return "\n".join(lines) + "\n"
