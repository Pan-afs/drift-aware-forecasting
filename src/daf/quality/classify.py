"""Decide whether a sudden change is a data error or a real shift (drift).

Every event found by :func:`daf.quality.drift.detect_events` is judged by
what the series does *after* the change, relative to the level the causal
baseline expected:

=============================  ======================================  ============
Evidence after the change      Interpretation                          Verdict
=============================  ======================================  ============
values freeze (often at 0)     stuck sensor / collection outage        data_error
level moves by ~10^k           unit or scaling change in the feed      data_error
too few points observed yet    a spike and a shift look the same       pending
values return to expected      transient spike, bad value              data_error
values stay at the new level   persistent change in the process        drift
=============================  ======================================  ============

When related series are available (other grid cells, indicators from the
same part of the economy) they act as witnesses. A real shift in the
underlying process usually shows up in several of them at once, while a
broken feed is usually isolated:

* persistent shift, peers moved too    -> drift, higher confidence
* persistent shift, no peer moved      -> inconclusive (could be a definition
  or reporting change in that one series)
* transient spike, peers spiked too    -> inconclusive (a real short-lived
  event such as a holiday, not a broken value)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from daf.quality.drift import ChangeEvent, Decomposition, DetectorConfig, decompose, detect_events

DATA_ERROR = "data_error"
DRIFT = "drift"
INCONCLUSIVE = "inconclusive"
PENDING = "pending"


@dataclass(frozen=True)
class Verdict:
    label: str
    reason: str
    confidence: float
    ts_start: pd.Timestamp
    ts_end: pd.Timestamp
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Peer:
    """A related series with its own seasonal period (peers may differ in
    frequency from the target, e.g. weekly claims next to monthly payrolls)."""

    series: pd.Series
    period: int = 1


def _clean(s: pd.Series) -> pd.Series:
    return s[~s.index.duplicated(keep="last")].sort_index().astype(float)


def _assess(
    dec: Decomposition, event: ChangeEvent, config: DetectorConfig, period: int
) -> tuple[str, str, float, dict[str, Any]]:
    n = len(dec.values)
    t0 = event.start
    post_len = config.post_window(period)
    window = config.baseline_window(period)

    # Freeze the pre-change expectation at t0 and measure deviations from it.
    b0, slope0, scale0 = dec.baseline[t0], dec.slope[t0], dec.scale[t0]
    expected = b0 + slope0 * (np.arange(n) - t0)
    dev = dec.adjusted - expected

    anomalous = float(np.nanmedian(dev[event.start : event.end + 1]))
    post = dev[t0 : t0 + post_len]
    post = post[np.isfinite(post)]
    raw_post = dec.values[t0 : t0 + post_len]
    raw_post = raw_post[np.isfinite(raw_post)]
    raw_pre = dec.values[max(0, t0 - window) : t0]
    raw_pre = raw_pre[np.isfinite(raw_pre)]

    evidence: dict[str, Any] = {
        "peak_z": round(event.peak_z, 2),
        "jump_sigma": round(anomalous / scale0, 2),
        "post_points": int(len(post)),
    }

    # 1. Feed freezes after the change while it was moving before.
    if len(raw_post) >= 3 and np.ptp(raw_post) == 0 and len(raw_pre) >= 3 and np.ptp(raw_pre) > 0:
        what = "zero" if raw_post[0] == 0 else f"{raw_post[0]:g}"
        return DATA_ERROR, f"feed stuck at {what} after the change", 0.95, evidence

    # 2. Scaling by a power of ten (units changed, decimal point moved).
    # Raw levels are used here: rescaling also rescales the seasonal pattern,
    # which would distort the seasonally adjusted values.
    raw_event = dec.values[event.start : event.end + 1]
    pre_level = float(np.median(raw_pre)) if len(raw_pre) else np.nan
    event_level = float(np.nanmedian(raw_event)) if np.isfinite(raw_event).any() else np.nan
    if pre_level > 0 and event_level > 0:
        ratio = event_level / pre_level
        log_ratio = float(np.log10(ratio))
        k = int(round(log_ratio))
        if k != 0 and abs(log_ratio - k) <= config.unit_tolerance:
            evidence["ratio"] = round(ratio, 4)
            return (
                DATA_ERROR,
                f"level changed by a factor of ~10^{k} (x{ratio:.3g}): unit or scaling error",
                0.9,
                evidence,
            )

    # 3. Not enough history after the change to tell a spike from a shift.
    available = n - t0
    if available < post_len:
        return (
            PENDING,
            f"only {available} of {post_len} post-change points observed so far",
            0.0,
            evidence,
        )
    if len(post) < post_len / 2:
        return INCONCLUSIVE, "too many missing values after the change", 0.3, evidence

    # 4. Does the series stay at the anomalous level or return to expected?
    stays = np.abs(post - anomalous) < np.abs(post)
    persistence = float(stays.mean())
    shift_sigma = float(np.median(post) / scale0)
    evidence.update(persistence=round(persistence, 2), shift_sigma=round(shift_sigma, 2))

    if persistence < config.persistence_min:
        length = event.end - event.start + 1
        return (
            DATA_ERROR,
            f"transient: {length} anomalous point(s), then back to the expected level",
            round(min(0.95, 1.0 - persistence), 2),
            evidence,
        )
    return (
        DRIFT,
        f"level shifted by {shift_sigma:+.1f} sigma and held for "
        f"{persistence:.0%} of the next {post_len} points",
        round(min(0.9, 0.4 + 0.5 * persistence), 2),
        evidence,
    )


def _peer_onsets(
    peers: Mapping[str, pd.Series | Peer], period: int, config: DetectorConfig
) -> dict[str, list[pd.Timestamp]]:
    onsets = {}
    for name, peer in peers.items():
        if not isinstance(peer, Peer):
            peer = Peer(peer, period)
        series = _clean(peer.series)
        if series.notna().sum() < 3 * config.baseline_window(peer.period):
            continue  # too short to have a baseline; not a usable witness
        dec = decompose(series, peer.period, config)
        onsets[name] = [dec.index[e.start] for e in detect_events(dec, config, peer.period)]
    return onsets


def _with_peers(
    verdict: Verdict,
    peer_onsets: dict[str, list[pd.Timestamp]],
    tolerance: pd.Timedelta,
    config: DetectorConfig,
) -> Verdict:
    if not peer_onsets or verdict.label in (PENDING, INCONCLUSIVE):
        return verdict
    if verdict.label == DATA_ERROR and not verdict.reason.startswith("transient"):
        return verdict  # stuck feeds and unit errors are not explained by peers

    hits = sorted(
        name
        for name, onsets in peer_onsets.items()
        if any(abs(t - verdict.ts_start) <= tolerance for t in onsets)
    )
    support = len(hits) / len(peer_onsets)
    evidence = {**verdict.evidence, "peer_support": round(support, 2), "peers_moved": hits}
    seen = f"{len(hits)}/{len(peer_onsets)} peer series"

    if verdict.label == DATA_ERROR:
        if support < config.peer_min_support:
            return replace(verdict, evidence=evidence)
        return Verdict(
            INCONCLUSIVE,
            f"short-lived change also seen in {seen}: likely a real event, not a bad value",
            round(0.4 + 0.4 * support, 2),
            verdict.ts_start,
            verdict.ts_end,
            evidence,
        )

    persistence = verdict.evidence.get("persistence", 1.0)
    if support >= config.peer_min_support:
        return Verdict(
            DRIFT,
            f"{verdict.reason}; {seen} shifted at the same time",
            round(min(0.99, 0.5 + 0.25 * persistence + 0.25 * support), 2),
            verdict.ts_start,
            verdict.ts_end,
            evidence,
        )
    if not hits:
        return Verdict(
            INCONCLUSIVE,
            f"{verdict.reason}, but none of the {len(peer_onsets)} peer series moved: "
            "possible definition or reporting change",
            0.5,
            verdict.ts_start,
            verdict.ts_end,
            evidence,
        )
    return replace(verdict, reason=f"{verdict.reason}; {seen} moved", evidence=evidence)


def classify_changes(
    s: pd.Series,
    period: int = 1,
    config: DetectorConfig | None = None,
    peers: Mapping[str, pd.Series | Peer] | None = None,
) -> list[Verdict]:
    """Find sudden changes in ``s`` and label each one.

    ``s`` should be on a regular grid (missing timestamps as NaN) so that
    ``period`` lines up with the seasonal cycle. ``peers`` are related series
    used to corroborate or question each verdict; a plain Series is assumed
    to share the target's period.
    """
    config = config or DetectorConfig()
    s = _clean(s)
    dec = decompose(s, period, config)
    events = detect_events(dec, config, period)
    refractory = config.baseline_window(period) // 2

    peer_onsets = _peer_onsets(peers, period, config) if peers else {}
    step = pd.Series(s.index).diff().median() if len(s) > 1 else pd.Timedelta(0)
    tolerance = config.peer_tolerance * step

    verdicts: list[Verdict] = []
    last_shift_end: int | None = None
    for event in events:
        # Right after a real shift the baseline is still catching up and may
        # overshoot; those echoes belong to the shift that caused them.
        if last_shift_end is not None and event.start - last_shift_end <= refractory:
            prev = verdicts[-1]
            verdicts[-1] = Verdict(
                prev.label, prev.reason, prev.confidence, prev.ts_start,
                dec.index[event.end], prev.evidence,
            )
            last_shift_end = event.end
            continue

        label, reason, confidence, evidence = _assess(dec, event, config, period)
        verdict = Verdict(
            label, reason, confidence, dec.index[event.start], dec.index[event.end], evidence
        )
        verdicts.append(_with_peers(verdict, peer_onsets, tolerance, config))
        last_shift_end = event.end if label in (DRIFT, INCONCLUSIVE) else None
    return verdicts
