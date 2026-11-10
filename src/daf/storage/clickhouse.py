"""Thin wrapper around clickhouse-connect for the tables in clickhouse/init."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from daf.config import ClickHouseSettings

OBS_COLUMNS = ["source", "series_id", "ts", "value"]
META_COLUMNS = ["source", "series_id", "title", "frequency", "units", "seasonal_period"]
FLAG_COLUMNS = [
    "source",
    "series_id",
    "ts_start",
    "ts_end",
    "check_name",
    "verdict",
    "confidence",
    "reason",
    "evidence",
    "run_id",
]


class ClickHouseStore:
    """Read/write helpers for observations and series metadata.

    The client is created lazily so the class can be imported (e.g. at DAG
    parse time) without a running server.
    """

    def __init__(self, settings: ClickHouseSettings | None = None, client: Any = None):
        self.settings = settings or ClickHouseSettings.from_env()
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import clickhouse_connect

            s = self.settings
            self._client = clickhouse_connect.get_client(
                host=s.host, port=s.port, username=s.user, password=s.password, database=s.database
            )
        return self._client

    # -- observations ---------------------------------------------------

    def insert_observations(self, frame: pd.DataFrame, chunk_size: int = 100_000) -> int:
        """Insert long-format rows (source, series_id, ts, value). NaN is stored as NULL."""
        if frame.empty:
            return 0
        missing = set(OBS_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"observations frame is missing columns: {sorted(missing)}")

        ts = pd.to_datetime(frame["ts"], utc=True)
        values = frame["value"].astype("float64")
        rows = [
            (src, sid, t.to_pydatetime(), None if pd.isna(v) else float(v))
            for src, sid, t, v in zip(frame["source"], frame["series_id"], ts, values)
        ]
        for i in range(0, len(rows), chunk_size):
            self.client.insert("observations", rows[i : i + chunk_size], column_names=OBS_COLUMNS)
        return len(rows)

    def latest_ts(self, source: str, series_id: str) -> pd.Timestamp | None:
        result = self.client.query(
            "SELECT count(), max(ts) FROM observations "
            "WHERE source = {source:String} AND series_id = {series_id:String}",
            parameters={"source": source, "series_id": series_id},
        )
        count, latest = result.result_rows[0]
        if not count:
            return None
        ts = pd.Timestamp(latest)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    def fetch_series(
        self, source: str, series_id: str, start: str | pd.Timestamp | None = None
    ) -> pd.Series:
        """Deduplicated series indexed by UTC timestamp (missing values as NaN)."""
        sql = (
            "SELECT ts, value FROM observations FINAL "
            "WHERE source = {source:String} AND series_id = {series_id:String}"
        )
        params: dict[str, Any] = {"source": source, "series_id": series_id}
        if start is not None:
            sql += " AND ts >= parseDateTime64BestEffort({start:String}, 3, 'UTC')"
            params["start"] = pd.Timestamp(start).isoformat()
        sql += " ORDER BY ts"

        df = self.client.query_df(sql, parameters=params)
        index = pd.DatetimeIndex(pd.to_datetime(df["ts"], utc=True), name="ts")
        return pd.Series(df["value"].to_numpy(dtype="float64"), index=index, name=series_id)

    def list_series(self, source: str) -> list[str]:
        result = self.client.query(
            "SELECT DISTINCT series_id FROM observations WHERE source = {source:String} "
            "ORDER BY series_id",
            parameters={"source": source},
        )
        return [row[0] for row in result.result_rows]

    # -- metadata -------------------------------------------------------

    def upsert_series_meta(self, rows: Iterable[Mapping[str, Any]]) -> int:
        data = [[row.get(col, "") for col in META_COLUMNS] for row in rows]
        if data:
            self.client.insert("series_meta", data, column_names=META_COLUMNS)
        return len(data)

    # -- quality flags --------------------------------------------------

    def replace_flags(
        self, source: str, series_id: str, flags: pd.DataFrame, run_id: str = ""
    ) -> int:
        """Store the current assessment of a series, dropping its older flags.

        Each run re-assesses the full history, so earlier flags (e.g. a
        'pending' verdict that has since resolved) are removed first.
        """
        self.client.command(
            "DELETE FROM quality_flags WHERE source = {source:String} "
            "AND series_id = {series_id:String}",
            parameters={"source": source, "series_id": series_id},
        )
        if flags.empty:
            return 0
        rows = [
            (
                source,
                series_id,
                pd.Timestamp(r.ts_start).to_pydatetime(),
                pd.Timestamp(r.ts_end).to_pydatetime(),
                r.check_name,
                r.verdict,
                float(r.confidence),
                r.reason,
                r.evidence,
                run_id,
            )
            for r in flags.itertuples(index=False)
        ]
        self.client.insert("quality_flags", rows, column_names=FLAG_COLUMNS)
        return len(rows)

    def get_series_meta(self, source: str) -> pd.DataFrame:
        return self.client.query_df(
            "SELECT source, series_id, title, frequency, units, seasonal_period "
            "FROM series_meta FINAL WHERE source = {source:String} ORDER BY series_id",
            parameters={"source": source},
        )
