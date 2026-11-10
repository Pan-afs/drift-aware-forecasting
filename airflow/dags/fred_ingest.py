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

    # Runs after every series is loaded, so peers are up to date.
    @task(max_active_tis_per_dagrun=4)
    def check_quality(result: dict) -> int:
        from airflow.operators.python import get_current_context

        from daf.config import Settings
        from daf.ingest.fred import PANDAS_FREQ
        from daf.quality.runner import peers_for, run_quality
        from daf.storage import ClickHouseStore

        series_id = result["series_id"]
        return run_quality(
            ClickHouseStore(Settings.from_env().clickhouse),
            "fred",
            series_id,
            peer_ids=peers_for(series_id, FRED.get("peer_groups", {})),
            run_id=get_current_context()["run_id"],
            freq_of=PANDAS_FREQ.get,
            period=result["seasonal_period"],
            freq=PANDAS_FREQ.get(result["frequency"]),
            # Daily FRED series mark market holidays as missing; not an error.
            check_missing=result["frequency"] != "D",
            # Spreads and growth rates can be negative. No flatline check either:
            # policy rates can legitimately sit still for years.
            non_negative=False,
        )

    check_quality.expand(result=ingest_series.expand(series_id=SERIES_IDS))


fred_ingest()
