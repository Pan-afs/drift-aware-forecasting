from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from daf.forecasting.metrics import (
    DEFAULT_QUANTILES,
    gmean,
    mase,
    median_index,
    pinball_loss,
    seasonal_error,
    wql,
)
from daf.forecasting.model import ChronosBolt, SeasonalNaive
from daf.forecasting.windows import rolling_windows

Q = DEFAULT_QUANTILES


# -- metrics -----------------------------------------------------------------


def test_seasonal_error():
    assert seasonal_error(np.arange(10.0)) == 1.0
    # a perfectly periodic context has no seasonal-naive error to scale by
    assert np.isnan(seasonal_error(np.tile([0.0, 5.0], 5), season=2))
    # season longer than the context falls back to lag 1
    assert seasonal_error(np.array([0.0, 2.0, 4.0]), season=10) == 2.0


def test_mase_perfect_and_naive():
    context = np.arange(20.0)
    y = np.array([20.0, 21.0, 22.0])
    assert mase(y, y, context) == 0.0
    # predicting the last value: errors 1, 2, 3 vs in-sample naive error 1
    assert mase(y, np.full(3, 19.0), context) == pytest.approx(2.0)


def test_mase_ignores_missing_targets():
    context = np.arange(20.0)
    y = np.array([20.0, np.nan, 22.0])
    assert mase(y, np.array([21.0, 0.0, 22.0]), context) == pytest.approx(0.5)


def test_pinball_is_asymmetric():
    assert pinball_loss(10.0, 8.0, 0.9) == pytest.approx(1.8)
    assert pinball_loss(10.0, 12.0, 0.9) == pytest.approx(0.2)


def test_wql_zero_for_perfect_quantiles_and_scale_free():
    y = np.array([1.0, 2.0, 3.0])
    assert wql(y, np.tile(y, (len(Q), 1)), Q) == 0.0
    noisy = np.tile(y, (len(Q), 1)) + np.linspace(-1, 1, len(Q))[:, None]
    assert wql(10 * y, 10 * noisy, Q) == pytest.approx(wql(y, noisy, Q))


def test_wql_checks_shape():
    with pytest.raises(ValueError):
        wql(np.ones(3), np.ones((2, 3)), Q)


def test_median_index_and_gmean():
    assert median_index(Q) == 4
    with pytest.raises(ValueError):
        median_index([0.1, 0.9])
    assert gmean([1.0, 4.0, np.nan, -1.0]) == pytest.approx(2.0)


# -- windows -----------------------------------------------------------------


def _series(n=100):
    return pd.Series(np.arange(n, dtype=float), index=pd.date_range("2024", periods=n, freq="h"))


def test_rolling_windows_layout():
    windows = rolling_windows(
        {"a": _series()}, prediction_length=10, n_windows=3, context_length=30
    )

    assert [w.target[0] for w in windows] == [70.0, 80.0, 90.0]
    assert windows[-1].target[-1] == 99.0
    for w in windows:
        assert len(w.context) == 30
        assert w.context[-1] + 1 == w.target[0]  # context ends right before origin


def test_rolling_windows_skip_short_series():
    assert rolling_windows({"a": _series(15)}, prediction_length=10) == []


# -- forecasters -------------------------------------------------------------


def test_seasonal_naive_repeats_last_season():
    x = np.tile(np.arange(24.0), 5)
    q = SeasonalNaive(24).predict_quantiles([x], 30, Q)
    assert q.shape == (1, len(Q), 30)
    np.testing.assert_allclose(q[0, median_index(Q)], np.resize(np.arange(24.0), 30))


def test_seasonal_naive_handles_nan_and_short_context():
    q = SeasonalNaive(24).predict_quantiles([np.array([1.0, np.nan, 3.0])], 4, Q)
    assert np.isfinite(q).all()


def _fake_pipeline(context_length=8):
    config = SimpleNamespace(input_patch_size=4, context_length=context_length)
    return SimpleNamespace(model=SimpleNamespace(chronos_config=config))


def test_chronos_prepare_truncates_and_transforms():
    model = ChronosBolt(pipeline=_fake_pipeline(), context_transform=lambda c: c[::2])
    (prepared,) = model.prepare([np.arange(20.0)])
    assert model.patch_size == 4
    np.testing.assert_array_equal(prepared, [12.0, 14.0, 16.0, 18.0])
    assert prepared.dtype == np.float32
