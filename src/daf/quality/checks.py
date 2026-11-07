"""Deterministic validity checks that need no model of the series.

They catch the unambiguous data errors: missing values, holes in the
timestamp grid, duplicated timestamps, impossible negative values and feeds
that are stuck on one value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Issue:
    check: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_points: int
    detail: str


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of consecutive True values."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(s), int(e) - 1) for s, e in zip(edges[::2], edges[1::2])]


def missing_values(s: pd.Series) -> list[Issue]:
    return [
        Issue("missing_value", s.index[a], s.index[b], b - a + 1, f"{b - a + 1} missing value(s)")
        for a, b in runs(s.isna().to_numpy())
    ]


def duplicate_timestamps(s: pd.Series) -> list[Issue]:
    dup = s.index[s.index.duplicated()].unique()
    return [Issue("duplicate_timestamp", t, t, 1, "timestamp appears more than once") for t in dup]


def timestamp_gaps(s: pd.Series, freq: str | None = None, max_step_ratio: float = 1.5) -> list[Issue]:
    """Timestamps absent from the expected grid.

    With ``freq`` the grid is explicit (``pd.date_range``). Without it a gap is
    any step longer than ``max_step_ratio`` times the median step, which works
    for regular series whose anchor is unknown (e.g. weekly FRED data).
    """
    index = s.index.unique().sort_values()
    if len(index) < 3:
        return []

    if freq is not None:
        expected = pd.date_range(index[0], index[-1], freq=freq)
        absent = expected.difference(index)
        if absent.empty:
            return []
        mask = expected.isin(absent)
        return [
            Issue("timestamp_gap", expected[a], expected[b], b - a + 1,
                  f"{b - a + 1} timestamp(s) missing from {freq} grid")
            for a, b in runs(mask)
        ]

    steps = np.diff(index.asi8)
    median = np.median(steps)
    issues = []
    for i in np.flatnonzero(steps > max_step_ratio * median):
        n_missing = int(round(steps[i] / median)) - 1
        issues.append(
            Issue("timestamp_gap", index[i], index[i + 1], max(n_missing, 1),
                  f"step of {pd.Timedelta(int(steps[i]))} vs typical {pd.Timedelta(int(median))}")
        )
    return issues


def negative_values(s: pd.Series) -> list[Issue]:
    return [
        Issue("negative_value", s.index[a], s.index[b], b - a + 1,
              f"{b - a + 1} negative value(s) in a non-negative series")
        for a, b in runs((s < 0).to_numpy())
    ]


def flatlines(s: pd.Series, min_run: int) -> list[Issue]:
    """Runs of at least ``min_run`` identical consecutive values (stuck feed)."""
    values = s.to_numpy(dtype=float)
    same = np.zeros(len(values), dtype=bool)
    if len(values) > 1:
        same[1:] = values[1:] == values[:-1]
    issues = []
    for a, b in runs(same):
        start = a - 1  # the run includes the first value that is repeated
        length = b - start + 1
        if length >= min_run:
            label = "stuck at zero" if values[start] == 0 else f"stuck at {values[start]:g}"
            issues.append(Issue("flatline", s.index[start], s.index[b], length,
                                f"{length} identical values ({label})"))
    return issues


def run_validity_checks(
    s: pd.Series,
    *,
    freq: str | None = None,
    non_negative: bool = True,
    flatline_run: int | None = None,
) -> list[Issue]:
    """All point-level checks. ``flatline_run=None`` disables the flatline check.

    Some series legitimately stay flat (policy rates between meetings), so the
    flatline length has to be chosen per source.
    """
    issues = duplicate_timestamps(s)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    issues += timestamp_gaps(s, freq=freq)
    issues += missing_values(s)
    if non_negative:
        issues += negative_values(s)
    if flatline_run:
        issues += flatlines(s.dropna(), flatline_run)
    return sorted(issues, key=lambda i: (i.start, i.check))
