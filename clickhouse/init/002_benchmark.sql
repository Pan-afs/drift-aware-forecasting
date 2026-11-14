-- Accuracy vs. speed results, one row per (benchmark run, model variant).
-- Init scripts only run on an empty volume; on an existing stack apply with
--   docker compose exec clickhouse clickhouse-client --multiquery < clickhouse/init/002_benchmark.sql
CREATE TABLE IF NOT EXISTS daf.benchmark_results
(
    run_id            String,
    created_at        DateTime64(3, 'UTC') DEFAULT now64(3),
    source            LowCardinality(String),
    model_id          LowCardinality(String),
    variant           LowCardinality(String),
    quantized         Bool,
    prune_strategy    LowCardinality(String),
    keep_ratio        Float32,
    context_length    UInt32,
    prediction_length UInt32,
    n_windows         UInt32,
    mase              Float64,
    wql               Float64,
    latency_ms_p50    Float64,
    latency_ms_p95    Float64,
    throughput        Float64,
    model_size_mb     Float64,
    encoder_tokens    Float64
)
ENGINE = MergeTree
ORDER BY (source, created_at, variant);
