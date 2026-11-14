import numpy as np
import pandas as pd
import pytest

from daf.forecasting.benchmark import (
    DEFAULT_VARIANTS,
    BenchmarkConfig,
    add_relative_columns,
    evaluate,
    load_csv,
    main,
    to_markdown,
)
from daf.forecasting.model import SeasonalNaive
from daf.forecasting.windows import rolling_windows

SEASON = 24


def _series(n_series=3, days=30):
    t = np.arange(24 * days)
    index = pd.date_range("2024-01-01", periods=len(t), freq="h")
    cycle = 100 + 10 * np.sin(2 * np.pi * t / SEASON)
    return {
        f"s{i}": pd.Series(cycle + np.random.default_rng(i).normal(0, 1, len(t)), index=index)
        for i in range(n_series)
    }


class Shifted:
    """Seasonal naive with a constant bias: strictly worse than the baseline."""

    name = "shifted"

    def predict_quantiles(self, contexts, prediction_length, quantiles):
        return SeasonalNaive(SEASON).predict_quantiles(contexts, prediction_length, quantiles) + 5


@pytest.fixture
def windows():
    return rolling_windows(_series(), prediction_length=24, n_windows=2, context_length=256)


def test_evaluate_reports_accuracy_and_speed(windows):
    cfg = BenchmarkConfig(season=SEASON, repeats=2, batch_size=4)
    result = evaluate(SeasonalNaive(SEASON), windows, cfg)

    assert set(result) == {"mase", "wql", "latency_ms_p50", "latency_ms_p95", "throughput"}
    # in-sample scale of the same naive forecast: MASE close to 1
    assert 0.7 < result["mase"] < 1.3
    assert result["latency_ms_p95"] >= result["latency_ms_p50"] > 0
    assert result["throughput"] > 0


def test_evaluate_ranks_a_worse_model_lower(windows):
    cfg = BenchmarkConfig(season=SEASON, repeats=1)
    good = evaluate(SeasonalNaive(SEASON), windows, cfg)
    bad = evaluate(Shifted(), windows, cfg)
    assert bad["mase"] > good["mase"]
    assert bad["wql"] > good["wql"]


def test_relative_columns_and_markdown():
    results = pd.DataFrame(
        [
            {"variant": "fp32", "mase": 1.0, "wql": 0.1, "latency_ms_p50": 100.0,
             "encoder_tokens": 32.0, "model_size_mb": 191.0},
            {"variant": "int8", "mase": 1.1, "wql": 0.1, "latency_ms_p50": 50.0,
             "encoder_tokens": 32.0, "model_size_mb": 60.0},
        ]
    )
    out = add_relative_columns(results)
    assert out.loc[1, "speedup"] == pytest.approx(2.0)
    assert out.loc[1, "mase_vs_ref"] == pytest.approx(1.1)

    md = to_markdown(out, title="demo")
    assert md.startswith("## demo")
    assert "| int8 | 1.100 | 0.100 | 50.0 | 2.00x | 32 | 60.0 |" in md


def test_relative_columns_without_reference():
    results = pd.DataFrame(
        [{"variant": "int8", "mase": 1.0, "wql": 0.1, "latency_ms_p50": 1.0}]
    )
    assert "speedup" not in add_relative_columns(results)


def test_load_csv_round_trip(tmp_path):
    frames = [
        pd.DataFrame({"series_id": sid, "ts": s.index, "value": s.to_numpy()})
        for sid, s in _series(2, days=2).items()
    ]
    path = tmp_path / "data.csv"
    pd.concat(frames).to_csv(path, index=False)

    loaded = load_csv(path)
    assert sorted(loaded) == ["s0", "s1"]
    assert len(loaded["s0"]) == 48


def test_variant_names_are_unique():
    names = [v.name for v in DEFAULT_VARIANTS]
    assert len(names) == len(set(names))
    assert names[0] == "fp32"


def test_cli_rejects_unknown_variant(tmp_path):
    path = tmp_path / "data.csv"
    s = _series(1, days=2)["s0"]
    frame = pd.DataFrame({"series_id": "s0", "ts": s.index, "value": s.to_numpy()})
    frame.to_csv(path, index=False)
    with pytest.raises(SystemExit):
        main(["--csv", str(path), "--variants", "fp64"])
