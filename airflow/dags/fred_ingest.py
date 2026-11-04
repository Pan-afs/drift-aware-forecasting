"""Daily FRED ingestion: pull each configured series and upsert it into ClickHouse."""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task

from daf.config import load_sources

FRED = load_sources()["fred"]
SERIES_IDS = [s["id"] for s in FRED["series"]]


@dag(
    dag_id="fred_ingest",
    schedule="0 6 * * *",
    start_date=pendulum.datetime(2026, 11, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": pendulum.duration(minutes=5)},
    tags=["ingest", "fred"],
)
def fred_ingest():
    @task(max_active_tis_per_dagrun=4)
    def ingest_series(series_id: str) -> dict:
        from daf.config import Settings
        from daf.ingest.fred import FredClient, incremental_start
        from daf.storage import ClickHouseStore

        settings = Settings.from_env()
        client = FredClient(settings.fred_api_key)
        store = ClickHouseStore(settings.clickhouse)

        info = client.series_info(series_id)
        store.upsert_series_meta([info.as_meta()])

        start = incremental_start(
            store.latest_ts("fred", series_id),
            lookback_days=FRED.get("revision_lookback_days", 400),
            default_start=FRED.get("start", "1990-01-01"),
        )
        rows = store.insert_observations(client.observations(series_id, start=start))
        return {
            "series_id": series_id,
            "frequency": info.frequency,
            "seasonal_period": info.seasonal_period,
            "start": start,
            "rows": rows,
        }

    ingest_series.expand(series_id=SERIES_IDS)


fred_ingest()
