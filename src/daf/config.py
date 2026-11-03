"""Runtime settings, read from environment variables with local-dev defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ClickHouseSettings:
    host: str = "localhost"
    port: int = 8123
    user: str = "daf"
    password: str = "daf"
    database: str = "daf"

    @classmethod
    def from_env(cls) -> ClickHouseSettings:
        return cls(
            host=os.getenv("CLICKHOUSE_HOST", cls.host),
            port=int(os.getenv("CLICKHOUSE_PORT", cls.port)),
            user=os.getenv("CLICKHOUSE_USER", cls.user),
            password=os.getenv("CLICKHOUSE_PASSWORD", cls.password),
            database=os.getenv("CLICKHOUSE_DB", cls.database),
        )


@dataclass(frozen=True)
class Settings:
    clickhouse: ClickHouseSettings
    fred_api_key: str | None
    dataverse_api_token: str | None
    data_dir: Path
    config_dir: Path

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            clickhouse=ClickHouseSettings.from_env(),
            fred_api_key=os.getenv("FRED_API_KEY") or None,
            dataverse_api_token=os.getenv("DATAVERSE_API_TOKEN") or None,
            data_dir=Path(os.getenv("DAF_DATA_DIR", REPO_ROOT / "data")),
            config_dir=Path(os.getenv("DAF_CONFIG_DIR", REPO_ROOT / "configs")),
        )


def load_sources(config_dir: Path | None = None) -> dict[str, Any]:
    """Load the per-source configuration (series lists, date ranges, peers)."""
    path = (config_dir or Settings.from_env().config_dir) / "sources.yaml"
    with path.open() as fh:
        return yaml.safe_load(fh) or {}
