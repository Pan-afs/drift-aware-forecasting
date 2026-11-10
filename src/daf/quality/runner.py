"""Run every quality check for one series and turn the results into flag rows."""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from daf.quality.checks import run_validity_checks
from daf.quality.classify import DATA_ERROR, Peer, classify_changes
from daf.quality.drift import DetectorConfig

FLAG_COLUMNS = ["ts_start", "ts_end", "check_name", "verdict", "confidence", "reason", "evidence"]


def regularize(s: pd.Series, freq: str | None) -> pd.Series:
    """Deduplicate, sort and (if ``freq`` is known) put ``s`` on a full grid."""
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if freq and len(s) > 1:
        s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq=freq))
    return s


def peers_for(series_id: str, groups: Mapping[str, Sequence[str]]) -> list[str]:
    """All series sharing at least one peer group with ``series_id``."""
    peers: set[str] = set()
    for members in groups.values():
        if series_id in members:
            peers.update(members)
    peers.discard(series_id)
    return sorted(peers)


def assess_series(
    s: pd.Series,
    *,
    period: int = 1,
    freq: str | None = None,
    peers: Mapping[str, pd.Series | Peer] | None = None,
    non_negative: bool = True,
    check_missing: bool = True,
    flatline_run: int | None = None,
    config: DetectorConfig | None = None,
) -> pd.DataFrame:
    """Validity issues plus classified sudden changes, one row per finding."""
    rows: list[dict[str, Any]] = []

    for issue in run_validity_checks(
        s, freq=freq, non_negative=non_negative, flatline_run=flatline_run
    ):
        if issue.check == "missing_value" and not check_missing:
            continue
        rows.append(
            {
                "ts_start": issue.start,
                "ts_end": issue.end,
                "check_name": issue.check,
                "verdict": DATA_ERROR,
                "confidence": 1.0,
                "reason": issue.detail,
                "evidence": json.dumps({"n_points": issue.n_points}),
            }
        )

    # Plain Series peers share the target's grid; Peer objects bring their own.
    regular_peers = {
        name: p if isinstance(p, Peer) else regularize(p, freq) for name, p in (peers or {}).items()
    }
    for v in classify_changes(regularize(s, freq), period, config, regular_peers):
        rows.append(
            {
                "ts_start": v.ts_start,
                "ts_end": v.ts_end,
                "check_name": "sudden_change",
                "verdict": v.label,
                "confidence": v.confidence,
                "reason": v.reason,
                "evidence": json.dumps(v.evidence, default=str),
            }
        )
    return pd.DataFrame(rows, columns=FLAG_COLUMNS)


def run_quality(
    store: Any,
    source: str,
    series_id: str,
    *,
    peer_ids: Sequence[str] = (),
    run_id: str = "",
    freq_of: Callable[[str], str | None] | None = None,
    **options: Any,
) -> int:
    """Assess a stored series and replace its flags in ClickHouse.

    Each peer is put on its own grid using its ``series_meta`` row;
    ``freq_of`` maps a stored frequency code (e.g. FRED's "M") to a pandas
    frequency and defaults to using the code as is.
    """
    s = store.fetch_series(source, series_id)

    peers: dict[str, Peer] = {}
    wanted = [pid for pid in peer_ids if pid != series_id]
    if wanted:
        meta = store.get_series_meta(source).set_index("series_id")
        for pid in wanted:
            if pid not in meta.index:
                continue
            series = store.fetch_series(source, pid)
            if series.empty:
                continue
            code = meta.at[pid, "frequency"]
            freq = freq_of(code) if freq_of else code
            peers[pid] = Peer(regularize(series, freq), int(meta.at[pid, "seasonal_period"]))

    flags = assess_series(s, peers=peers, **options)
    return store.replace_flags(source, series_id, flags, run_id=run_id)
