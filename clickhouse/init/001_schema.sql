CREATE DATABASE IF NOT EXISTS daf;

-- Raw observations from every source, one row per (series, timestamp).
-- ReplacingMergeTree keeps the most recently ingested version of a row, so
-- re-pulling a window of FRED data picks up revisions without duplicates.
-- Query with FINAL (or argMax on ingested_at) to read deduplicated values.
CREATE TABLE IF NOT EXISTS daf.observations
(
    source      LowCardinality(String),
    series_id   LowCardinality(String),
    ts          DateTime64(3, 'UTC'),
    value       Nullable(Float64),
    ingested_at DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY source
ORDER BY (source, series_id, ts);

-- Descriptive metadata per series; seasonal_period drives quality checks
-- and the seasonal-naive scale in MASE.
CREATE TABLE IF NOT EXISTS daf.series_meta
(
    source          LowCardinality(String),
    series_id       LowCardinality(String),
    title           String,
    frequency       LowCardinality(String),
    units           String,
    seasonal_period UInt16,
    updated_at      DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (source, series_id);

-- One row per detected issue or change event. Re-running a check over the
-- same window replaces the earlier verdict (e.g. 'pending' -> 'drift').
CREATE TABLE IF NOT EXISTS daf.quality_flags
(
    source      LowCardinality(String),
    series_id   LowCardinality(String),
    ts_start    DateTime64(3, 'UTC'),
    ts_end      DateTime64(3, 'UTC'),
    check_name  LowCardinality(String),
    verdict     Enum8('ok' = 0, 'data_error' = 1, 'drift' = 2, 'inconclusive' = 3, 'pending' = 4),
    confidence  Float32,
    reason      String,
    evidence    String,
    run_id      String,
    detected_at DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(detected_at)
ORDER BY (source, series_id, check_name, ts_start);
