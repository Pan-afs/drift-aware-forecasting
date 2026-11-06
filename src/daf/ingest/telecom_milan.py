"""Loader for the Telecom Italia Big Data Challenge "Milan grid" dataset.

The SMS / Call / Internet activity files are published on Harvard Dataverse
(doi:10.7910/DVN/EGZHFV), one tab-separated file per day from 2013-11-01 to
2014-01-01. Each row is

    square_id, time_interval, country_code, sms_in, sms_out, call_in, call_out, internet

for one 10-minute interval (epoch milliseconds, UTC), one cell of a 100x100
grid over Milan and one caller country code. Activity values are anonymised,
rescaled counts; an empty field means no activity was recorded.
"""

from __future__ import annotations

import logging
import re
import zipfile
from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd
import requests

log = logging.getLogger(__name__)

SOURCE = "telecom_milan"
DATASET_DOI = "doi:10.7910/DVN/EGZHFV"
DATAVERSE_URL = "https://dataverse.harvard.edu"
FILE_PATTERN = re.compile(r"sms-call-internet-mi-(\d{4}-\d{2}-\d{2})")
FIRST_DAY = date(2013, 11, 1)
LAST_DAY = date(2014, 1, 1)

COLUMNS = [
    "square_id",
    "time_interval",
    "country_code",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet",
]
ACTIVITIES = ["sms_in", "sms_out", "call_in", "call_out", "internet"]
DTYPES = {
    "square_id": "int32",
    "time_interval": "int64",
    "country_code": "int32",
    **{a: "float32" for a in ACTIVITIES},
}
CITY = "city"


class DataverseClient:
    """Lists and downloads the daily activity files from Harvard Dataverse."""

    def __init__(
        self,
        base_url: str = DATAVERSE_URL,
        api_token: str | None = None,
        session: requests.Session | None = None,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        if api_token:
            self.session.headers["X-Dataverse-key"] = api_token

    def list_files(self, doi: str = DATASET_DOI) -> dict[date, int]:
        """Map each day in the dataset to its Dataverse file id."""
        resp = self.session.get(
            f"{self.base_url}/api/datasets/:persistentId/versions/:latest/files",
            params={"persistentId": doi},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        index: dict[date, int] = {}
        for entry in resp.json()["data"]:
            data_file = entry["dataFile"]
            name = data_file.get("originalFileName") or data_file.get("filename") or ""
            match = FILE_PATTERN.search(name)
            if match:
                index[date.fromisoformat(match.group(1))] = data_file["id"]
        return index

    def download(self, file_id: int, dest: Path) -> Path:
        """Stream a file to ``dest`` (written atomically via a .part file)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        with self.session.get(
            f"{self.base_url}/api/access/datafile/{file_id}",
            params={"format": "original"},
            stream=True,
            timeout=self.timeout,
        ) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        tmp.replace(dest)
        return dest


def find_cached(day: date, cache_dir: Path) -> Path | None:
    """A previously downloaded (or manually placed) file for ``day``, if any."""
    for path in sorted(cache_dir.glob(f"sms-call-internet-mi-{day.isoformat()}*")):
        if not path.name.endswith(".part"):
            return path
    return None


def fetch_day(
    day: date,
    cache_dir: Path,
    client: DataverseClient | None = None,
    file_index: dict[date, int] | None = None,
) -> Path:
    """Local path of the file for ``day``, downloading it on a cache miss.

    Files can also be downloaded by hand from Dataverse and dropped into
    ``cache_dir``; anything matching ``sms-call-internet-mi-YYYY-MM-DD*`` is used.
    """
    if not FIRST_DAY <= day <= LAST_DAY:
        raise ValueError(f"{day} is outside the dataset range {FIRST_DAY}..{LAST_DAY}")
    cached = find_cached(day, cache_dir)
    if cached is not None:
        return cached

    client = client or DataverseClient()
    index = file_index if file_index is not None else client.list_files()
    if day not in index:
        raise FileNotFoundError(f"no Dataverse file for {day}")
    log.info("downloading Telecom Italia file for %s", day)
    return client.download(index[day], cache_dir / f"sms-call-internet-mi-{day.isoformat()}.txt")


def _first_line(path: Path) -> str:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf, zf.open(zf.namelist()[0]) as fh:
            return fh.readline().decode("utf-8", errors="replace")
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return fh.readline()


def read_day_file(path: Path) -> pd.DataFrame:
    """Read one daily file (plain or zipped, with or without a header row)."""
    first = _first_line(path).split("\t", 1)[0].strip().strip('"')
    has_header = not first.isdigit()
    return pd.read_csv(
        path,
        sep="\t",
        header=0 if has_header else None,
        names=COLUMNS,
        dtype=DTYPES,
        compression="zip" if zipfile.is_zipfile(path) else None,
    )


def series_id(activity: str, area: str | int) -> str:
    """``internet/city`` for the whole grid, ``internet/sq5060`` for one cell."""
    return f"{activity}/{area if area == CITY else f'sq{int(area)}'}"


def build_series(
    raw: pd.DataFrame,
    squares: Sequence[int] = (),
    activities: Sequence[str] = ACTIVITIES,
    freq: str = "1h",
) -> pd.DataFrame:
    """Aggregate raw grid rows into long-format observations.

    Activity is summed over country codes, giving one series per activity for
    the whole city and one per selected grid cell, then resampled to ``freq``.
    Missing 10-minute intervals are *not* imputed: a hole in the raw feed shows
    up as NaN or as an undercounted bucket, which is exactly what the quality
    checks are meant to catch.
    """
    activities = list(activities)
    unknown = set(activities) - set(ACTIVITIES)
    if unknown:
        raise ValueError(f"unknown activities: {sorted(unknown)}")

    frame = raw[["square_id", "time_interval", *activities]].copy()
    frame["ts"] = pd.to_datetime(frame["time_interval"], unit="ms", utc=True)

    groups: dict[str | int, pd.DataFrame] = {
        CITY: frame.groupby("ts")[activities].sum(min_count=1)
    }
    cells = frame[frame["square_id"].isin(list(squares))]
    for square, sub in cells.groupby("square_id"):
        groups[int(square)] = sub.groupby("ts")[activities].sum(min_count=1)

    parts = []
    for area, wide in groups.items():
        wide = wide.astype("float64").resample(freq).sum(min_count=1)
        for activity in activities:
            parts.append(
                pd.DataFrame(
                    {
                        "source": SOURCE,
                        "series_id": series_id(activity, area),
                        "ts": wide.index,
                        "value": wide[activity].to_numpy(),
                    }
                )
            )
    if not parts:
        return pd.DataFrame(columns=["source", "series_id", "ts", "value"])
    return pd.concat(parts, ignore_index=True)
