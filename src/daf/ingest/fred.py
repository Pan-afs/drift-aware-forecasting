"""Client for the FRED (Federal Reserve Economic Data) REST API.

API docs: https://fred.stlouisfed.org/docs/api/fred/
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.stlouisfed.org/fred"
SOURCE = "fred"
RETRY_STATUS = {429, 500, 502, 503, 504}
OBS_COLUMNS = ["source", "series_id", "ts", "value"]

# FRED frequency code -> seasonal period used by quality checks and MASE.
SEASONAL_PERIOD = {"D": 1, "W": 52, "BW": 26, "M": 12, "Q": 4, "SA": 2, "A": 1}

# FRED frequency code -> pandas frequency of the expected timestamp grid.
# Daily FRED series are business-daily and list holidays as missing ("."),
# so "B" is the right grid. Weekly series have varying anchors (W-SAT, W-FRI),
# so they are left to step-based gap detection.
PANDAS_FREQ = {"D": "B", "M": "MS", "Q": "QS", "A": "YS"}


class FredError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeriesInfo:
    series_id: str
    title: str
    frequency: str
    units: str
    seasonal_adjustment: str
    last_updated: str

    @property
    def seasonal_period(self) -> int:
        return SEASONAL_PERIOD.get(self.frequency, 1)

    def as_meta(self) -> dict[str, Any]:
        return {
            "source": SOURCE,
            "series_id": self.series_id,
            "title": self.title,
            "frequency": self.frequency,
            "units": self.units,
            "seasonal_period": self.seasonal_period,
        }


class FredClient:
    def __init__(
        self,
        api_key: str | None,
        session: requests.Session | None = None,
        max_retries: int = 4,
        backoff: float = 1.0,
        timeout: float = 30.0,
    ):
        if not api_key:
            raise FredError(
                "FRED_API_KEY is not set; get a free key at "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self.api_key = api_key
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.backoff = backoff
        self.timeout = timeout

    def _get(self, endpoint: str, **params: Any) -> dict[str, Any]:
        query = {**params, "api_key": self.api_key, "file_type": "json"}
        url = f"{BASE_URL}/{endpoint}"
        for attempt in range(self.max_retries + 1):
            resp = self.session.get(url, params=query, timeout=self.timeout)
            if resp.status_code in RETRY_STATUS and attempt < self.max_retries:
                wait = self.backoff * 2**attempt
                log.warning("FRED %s returned %s, retrying in %.1fs", endpoint, resp.status_code, wait)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                raise FredError(f"FRED {endpoint} failed ({resp.status_code}): {resp.text[:200]}")
            return resp.json()
        raise FredError(f"FRED {endpoint} failed after {self.max_retries} retries")

    def series_info(self, series_id: str) -> SeriesInfo:
        payload = self._get("series", series_id=series_id)
        s = payload["seriess"][0]
        return SeriesInfo(
            series_id=s["id"],
            title=s["title"],
            frequency=s["frequency_short"],
            units=s["units"],
            seasonal_adjustment=s["seasonal_adjustment_short"],
            last_updated=s["last_updated"],
        )

    def observations(
        self,
        series_id: str,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        params: dict[str, Any] = {"series_id": series_id}
        if start is not None:
            params["observation_start"] = pd.Timestamp(start).strftime("%Y-%m-%d")
        if end is not None:
            params["observation_end"] = pd.Timestamp(end).strftime("%Y-%m-%d")
        return parse_observations(self._get("series/observations", **params), series_id)


def parse_observations(payload: dict[str, Any], series_id: str) -> pd.DataFrame:
    """Convert a FRED observations payload to long format.

    FRED encodes missing values as "." (holidays, discontinued months). They
    are kept as NaN rather than dropped so the quality checks can see the gap.
    """
    obs = payload.get("observations") or []
    if not obs:
        return pd.DataFrame(columns=OBS_COLUMNS)
    frame = pd.DataFrame(obs)
    values = pd.to_numeric(frame["value"].replace(".", np.nan), errors="coerce")
    return pd.DataFrame(
        {
            "source": SOURCE,
            "series_id": series_id,
            "ts": pd.to_datetime(frame["date"], utc=True),
            "value": values.astype("float64"),
        }
    )


def incremental_start(
    latest: pd.Timestamp | None, lookback_days: int, default_start: str
) -> str:
    """First date to request on an incremental run.

    FRED revises recent history (and benchmark revisions reach back further),
    so each run re-pulls a lookback window; ReplacingMergeTree keeps the newest
    version of every row.
    """
    if latest is None:
        return default_start
    start = max(latest - pd.Timedelta(days=lookback_days), pd.Timestamp(default_start, tz="UTC"))
    return start.strftime("%Y-%m-%d")
