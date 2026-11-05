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
from datetime import date
from pathlib import Path

import requests

log = logging.getLogger(__name__)

SOURCE = "telecom_milan"
DATASET_DOI = "doi:10.7910/DVN/EGZHFV"
DATAVERSE_URL = "https://dataverse.harvard.edu"
FILE_PATTERN = re.compile(r"sms-call-internet-mi-(\d{4}-\d{2}-\d{2})")
FIRST_DAY = date(2013, 11, 1)
LAST_DAY = date(2014, 1, 1)


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
