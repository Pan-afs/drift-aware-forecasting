import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from daf.ingest.telecom_milan import (
    COLUMNS,
    DataverseClient,
    build_series,
    fetch_day,
    find_cached,
    read_day_file,
    series_id,
)

# 2013-11-01 00:00 Europe/Rome == 2013-10-31 23:00 UTC
T0 = 1383260400000
TEN_MIN = 600_000


def _rows(n_intervals=12):
    rows = []
    for k in range(n_intervals):
        ts = T0 + k * TEN_MIN
        rows.append(f"5060\t{ts}\t39\t1.0\t\t2.0\t\t10.0")  # Italian callers
        rows.append(f"5060\t{ts}\t49\t0.5\t\t\t\t5.0")  # German callers, no calls
        rows.append(f"42\t{ts}\t39\t1.0\t\t\t\t1.0")
    return rows


@pytest.fixture
def day_file(tmp_path):
    path = tmp_path / "sms-call-internet-mi-2013-11-01.txt"
    path.write_text("\n".join(_rows()) + "\n")
    return path


def test_read_plain_file(day_file):
    raw = read_day_file(day_file)
    assert list(raw.columns) == COLUMNS
    assert len(raw) == 36
    assert raw["call_out"].isna().all()


def test_read_file_with_header_and_zip(tmp_path):
    text = "\t".join(COLUMNS) + "\n" + "\n".join(_rows(2)) + "\n"
    path = tmp_path / "sms-call-internet-mi-2013-11-01.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("sms-call-internet-mi-2013-11-01.txt", text)

    raw = read_day_file(path)
    assert len(raw) == 6
    assert raw["square_id"].iloc[0] == 5060


def test_build_series_sums_countries_and_resamples(day_file):
    out = build_series(read_day_file(day_file), squares=[5060], activities=["internet", "call_in"])

    cell = out[out["series_id"] == "internet/sq5060"].set_index("ts")["value"]
    city = out[out["series_id"] == "internet/city"].set_index("ts")["value"]
    # 6 ten-minute intervals per hour, (10 + 5) per interval for the cell
    assert cell.tolist() == [90.0, 90.0]
    assert city.tolist() == [96.0, 96.0]
    assert cell.index[0] == pd.Timestamp("2013-10-31 23:00", tz="UTC")
    assert set(out["series_id"]) == {
        "internet/city",
        "call_in/city",
        "internet/sq5060",
        "call_in/sq5060",
    }


def test_missing_hour_stays_missing(tmp_path):
    hour_two = range(T0 + 6 * TEN_MIN, T0 + 12 * TEN_MIN)
    rows = [r for r in _rows(18) if int(r.split("\t")[1]) not in hour_two]
    path = tmp_path / "f.txt"
    path.write_text("\n".join(rows) + "\n")

    out = build_series(read_day_file(path), activities=["internet"])
    values = out.set_index("ts")["value"]
    assert np.isnan(values.iloc[1])
    assert values.iloc[[0, 2]].tolist() == [96.0, 96.0]


def test_unknown_activity_rejected(day_file):
    with pytest.raises(ValueError):
        build_series(read_day_file(day_file), activities=["mms"])


def test_series_id():
    assert series_id("internet", "city") == "internet/city"
    assert series_id("sms_in", 5060) == "sms_in/sq5060"


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    headers: dict = {}

    def get(self, url, params=None, timeout=None, stream=False):
        return FakeResponse(
            {
                "data": [
                    {"dataFile": {"id": 11, "filename": "sms-call-internet-mi-2013-11-01.tab",
                                  "originalFileName": "sms-call-internet-mi-2013-11-01.txt"}},
                    {"dataFile": {"id": 12, "filename": "sms-call-internet-mi-2013-11-02.txt"}},
                    {"dataFile": {"id": 99, "filename": "README.pdf"}},
                ]
            }
        )


def test_list_files_maps_days_to_ids():
    index = DataverseClient(session=FakeSession()).list_files()
    assert index == {date(2013, 11, 1): 11, date(2013, 11, 2): 12}


def test_fetch_day_prefers_cache(day_file):
    assert find_cached(date(2013, 11, 1), day_file.parent) == day_file
    # client is never touched on a cache hit
    assert fetch_day(date(2013, 11, 1), day_file.parent, client=object()) == day_file


def test_fetch_day_rejects_out_of_range(tmp_path):
    with pytest.raises(ValueError):
        fetch_day(date(2015, 1, 1), tmp_path)
