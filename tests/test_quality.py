import json

import numpy as np
import pandas as pd
import pytest

from daf.quality import (
    DATA_ERROR,
    DRIFT,
    INCONCLUSIVE,
    PENDING,
    assess_series,
    classify_changes,
    run_validity_checks,
)
from daf.quality.checks import flatlines, runs, timestamp_gaps
from daf.quality.classify import Peer
from daf.quality.drift import DetectorConfig, cusum_alarms
from daf.quality.runner import peers_for, regularize

PERIOD = 24
CHANGE = 600


def make_series(n=24 * 40, seed=0, trend=0.0):
    """Hourly series with a daily cycle and unit-variance noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 100 + 10 * np.sin(2 * np.pi * t / PERIOD) + trend * t + rng.normal(0, 1, n)
    return pd.Series(y, index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))


def verdict_at(s, at=CHANGE, **kwargs):
    """The verdict whose event starts at the injected change."""
    ts = s.index[at]
    matches = [v for v in classify_changes(s, PERIOD, **kwargs) if v.ts_start == ts]
    assert matches, f"no event detected at {ts}"
    return matches[0]


# -- validity checks ---------------------------------------------------------


def test_runs():
    assert runs(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 2), (4, 4)]
    assert runs(np.zeros(3, dtype=bool)) == []


def test_validity_checks_find_each_problem():
    s = make_series(48)
    s.iloc[3] = np.nan
    s.iloc[10:16] = 0.0
    s.iloc[20] = -5.0
    s = pd.concat([s.drop(s.index[30:33]), s.iloc[[40]]])  # gap + duplicate

    found = {i.check for i in run_validity_checks(s, freq="h", flatline_run=4)}

    assert found == {
        "missing_value",
        "flatline",
        "negative_value",
        "timestamp_gap",
        "duplicate_timestamp",
    }


def test_gap_detection_without_frequency():
    s = make_series(30).drop(make_series(30).index[10:13])
    (gap,) = timestamp_gaps(s)
    assert gap.n_points == 3


def test_flatline_requires_min_run():
    s = make_series(20)
    s.iloc[5:8] = 1.0
    assert flatlines(s, 4) == []
    assert flatlines(s, 3)[0].n_points == 3


# -- change detection --------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_clean_series_has_no_events(seed):
    assert classify_changes(make_series(seed=seed), PERIOD) == []


def test_cusum_finds_onset_of_shift():
    z = np.r_[np.zeros(50), np.full(50, 2.0)]
    # A sustained shift keeps re-alarming after each reset; the first one counts.
    onset, alarm = cusum_alarms(z, k=0.75, h=10)[0]
    assert onset == 50
    assert 50 < alarm < 60


def test_spike_is_data_error():
    s = make_series()
    s.iloc[CHANGE] += 40
    v = verdict_at(s)
    assert v.label == DATA_ERROR
    assert v.reason.startswith("transient")


def test_level_shift_is_drift():
    s = make_series()
    s.iloc[CHANGE:] += 15
    v = verdict_at(s)
    assert v.label == DRIFT
    assert v.evidence["persistence"] > 0.9


def test_level_shift_on_trend_is_drift():
    s = make_series(trend=0.05)
    s.iloc[CHANGE:] -= 15
    assert verdict_at(s).label == DRIFT


def test_shift_is_reported_once():
    s = make_series()
    s.iloc[CHANGE:] += 15
    assert len([v for v in classify_changes(s, PERIOD) if v.label == DRIFT]) == 1


@pytest.mark.parametrize("factor, k", [(1000, 3), (0.01, -2)])
def test_power_of_ten_jump_is_unit_error(factor, k):
    s = make_series()
    s.iloc[CHANGE:] *= factor
    v = verdict_at(s)
    assert v.label == DATA_ERROR
    assert f"10^{k}" in v.reason


def test_drop_to_zero_is_stuck_feed():
    s = make_series()
    s.iloc[CHANGE : CHANGE + 50] = 0.0
    v = verdict_at(s)
    assert v.label == DATA_ERROR
    assert "stuck at zero" in v.reason


def test_change_at_the_end_is_pending():
    s = make_series()
    s.iloc[-3:] += 30
    v = verdict_at(s, at=len(s) - 3)
    assert v.label == PENDING


def test_config_controls_sensitivity():
    s = make_series()
    s.iloc[CHANGE] += 6
    assert classify_changes(s, PERIOD, DetectorConfig(z_threshold=4.0, cusum_h=50))
    assert not classify_changes(s, PERIOD, DetectorConfig(z_threshold=8.0, cusum_h=50))


# -- peers -------------------------------------------------------------------


def make_peers(shift=0.0, spike=0.0):
    peers = {}
    for seed in (11, 12, 13):
        p = make_series(seed=seed)
        p.iloc[CHANGE + 1 :] += shift
        p.iloc[CHANGE] += spike
        peers[f"peer{seed}"] = p
    return peers


def test_shift_confirmed_by_peers():
    s = make_series()
    s.iloc[CHANGE:] += 15
    v = verdict_at(s, peers=make_peers(shift=12))
    assert v.label == DRIFT
    assert v.evidence["peer_support"] == 1.0


def test_isolated_shift_is_questioned():
    s = make_series()
    s.iloc[CHANGE:] += 15
    v = verdict_at(s, peers=make_peers())
    assert v.label == INCONCLUSIVE
    assert "none of the 3 peer series moved" in v.reason


def test_spike_shared_with_peers_is_not_called_an_error():
    s = make_series()
    s.iloc[CHANGE] += 40
    assert verdict_at(s, peers=make_peers(spike=40)).label == INCONCLUSIVE
    assert verdict_at(s, peers=make_peers()).label == DATA_ERROR


def test_peers_with_their_own_period():
    s = make_series()
    s.iloc[CHANGE:] += 15
    # Peer sampled every 2 hours with a 12-step cycle.
    peer = make_series(seed=5)
    peer.iloc[CHANGE:] += 12
    peer = peer.iloc[::2]
    v = verdict_at(s, peers={"coarse": Peer(peer, period=12)})
    assert v.label == DRIFT


# -- runner ------------------------------------------------------------------


def test_peers_for():
    groups = {"a": ["X", "Y"], "b": ["Y", "Z"]}
    assert peers_for("Y", groups) == ["X", "Z"]
    assert peers_for("Q", groups) == []


def test_regularize_fills_grid():
    s = make_series(10).drop(make_series(10).index[4])
    out = regularize(s, "h")
    assert len(out) == 10
    assert np.isnan(out.iloc[4])


def test_assess_series_rows():
    s = make_series()
    s.iloc[CHANGE] += 40
    s.iloc[100] = np.nan

    flags = assess_series(s, period=PERIOD, freq="h")

    assert set(flags["check_name"]) == {"missing_value", "sudden_change"}
    assert set(flags["verdict"]) == {DATA_ERROR}
    for evidence in flags["evidence"]:
        json.loads(evidence)

    quiet = assess_series(s, period=PERIOD, freq="h", check_missing=False)
    assert "missing_value" not in set(quiet["check_name"])
