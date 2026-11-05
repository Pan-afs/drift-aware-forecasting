import numpy as np
import pandas as pd
import pytest

from daf.ingest import fred
from daf.ingest.fred import FredClient, FredError, incremental_start, parse_observations


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.responses.pop(0)


PAYLOAD = {
    "observations": [
        {"date": "2020-01-01", "value": "3.5"},
        {"date": "2020-02-01", "value": "."},
        {"date": "2020-03-01", "value": "4.4"},
    ]
}


def test_parse_keeps_missing_values_as_nan():
    frame = parse_observations(PAYLOAD, "UNRATE")

    assert list(frame.columns) == ["source", "series_id", "ts", "value"]
    assert frame["value"].dtype == np.float64
    assert np.isnan(frame["value"].iloc[1])
    assert frame["value"].iloc[2] == pytest.approx(4.4)
    assert str(frame["ts"].dt.tz) == "UTC"
    assert (frame["source"] == "fred").all()


def test_parse_empty_payload():
    frame = parse_observations({"observations": []}, "UNRATE")
    assert frame.empty
    assert list(frame.columns) == ["source", "series_id", "ts", "value"]


def test_client_requires_api_key():
    with pytest.raises(FredError):
        FredClient(None)


def test_client_retries_on_rate_limit(monkeypatch):
    monkeypatch.setattr(fred.time, "sleep", lambda _: None)
    session = FakeSession([FakeResponse(429), FakeResponse(200, PAYLOAD)])
    client = FredClient("key", session=session)

    frame = client.observations("UNRATE", start="2020-01-01")

    assert len(session.calls) == 2
    assert session.calls[0][1]["observation_start"] == "2020-01-01"
    assert session.calls[0][1]["file_type"] == "json"
    assert len(frame) == 3


def test_client_raises_on_client_error():
    session = FakeSession([FakeResponse(400, {"error_message": "Bad Request"})])
    with pytest.raises(FredError):
        FredClient("key", session=session).observations("NOPE")


def test_incremental_start():
    assert incremental_start(None, 400, "1990-01-01") == "1990-01-01"
    latest = pd.Timestamp("2026-10-01", tz="UTC")
    assert incremental_start(latest, 30, "1990-01-01") == "2026-09-01"
    # never earlier than the configured start
    assert incremental_start(latest, 100_000, "1990-01-01") == "1990-01-01"
