"""Backfill the Telecom Italia Milan dataset into ClickHouse and check its quality.

The dataset is static (Nov 2013 - Jan 2014), so the DAG is triggered manually
with an optional date range instead of running on a schedule.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.operators.python import get_current_context

from daf.config import load_sources
from daf.ingest.telecom_milan import CITY, SOURCE, series_id

MILAN = load_sources()["telecom_milan"]
SERIES_IDS = [
    series_id(activity, area)
    for activity in MILAN["activities"]
    for area in [CITY, *MILAN["squares"]]
]


@dag(
    dag_id="telecom_milan_ingest",
    schedule=None,
    start_date=pendulum.datetime(2026, 11, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": pendulum.duration(minutes=2)},
    params={
        "start": Param(MILAN["start"], type="string", format="date"),
        "end": Param(MILAN["end"], type="string", format="date"),
    },
    tags=["ingest", "telecom"],
)
def telecom_milan_ingest():
    @task
    def list_days() -> list[str]:
        import pandas as pd

        params = get_current_context()["params"]
        return [d.date().isoformat() for d in pd.date_range(params["start"], params["end"])]

    # Each raw file is a few hundred MB, so keep parallelism low.
    @task(max_active_tis_per_dagrun=2)
    def ingest_day(day: str) -> int:
        from datetime import date

        from daf.config import Settings
        from daf.ingest.telecom_milan import (
            DataverseClient,
            build_series,
            fetch_day,
            read_day_file,
        )
        from daf.storage import ClickHouseStore

        settings = Settings.from_env()
        client = DataverseClient(api_token=settings.dataverse_api_token)
        cache_dir = settings.data_dir / "raw" / "telecom_milan"
        path = fetch_day(date.fromisoformat(day), cache_dir, client)

        frame = build_series(
            read_day_file(path),
            squares=MILAN["squares"],
            activities=MILAN["activities"],
            freq=MILAN["freq"],
        )
        return ClickHouseStore(settings.clickhouse).insert_observations(frame)

    @task
    def register_series() -> list[str]:
        from daf.config import Settings
        from daf.storage import ClickHouseStore

        ClickHouseStore(Settings.from_env().clickhouse).upsert_series_meta(
            {
                "source": SOURCE,
                "series_id": sid,
                "title": f"Milan {sid.replace('/', ' ')} activity",
                "frequency": MILAN["freq"],
                "units": "rescaled activity count",
                "seasonal_period": MILAN["seasonal_period"],
            }
            for sid in SERIES_IDS
        )
        return SERIES_IDS

    @task(max_active_tis_per_dagrun=4)
    def check_quality(sid: str) -> int:
        from daf.config import Settings
        from daf.quality.runner import run_quality
        from daf.storage import ClickHouseStore

        # Peers: every other area for the same activity. A real change in how
        # people use the network (holidays, events) hits many cells at once.
        activity = sid.split("/", 1)[0]
        peers = [p for p in SERIES_IDS if p.startswith(f"{activity}/") and p != sid]
        return run_quality(
            ClickHouseStore(Settings.from_env().clickhouse),
            SOURCE,
            sid,
            peer_ids=peers,
            run_id=get_current_context()["run_id"],
            period=MILAN["seasonal_period"],
            freq=MILAN["freq"],
            flatline_run=MILAN.get("flatline_run"),
        )

    ids = register_series()
    ingest_day.expand(day=list_days()) >> ids
    check_quality.expand(sid=ids)


telecom_milan_ingest()
