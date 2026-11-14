# Drift-Aware Forecasting

A reproducible, end-to-end time-series system that ingests open data, checks its
quality, and forecasts it with a small, compressed foundation model.

## What it does
- **Pipeline**: Apache Airflow orchestrates ingestion into ClickHouse. Everything
  runs locally with `docker compose up`.
- **Data quality**: when a series changes suddenly, the system decides whether
  it is a *data error* (bad value, gap, unit or reporting glitch) or a
  *real shift* (drift) in the underlying process, and flags it.
- **Forecasting**: Chronos-Bolt with int8 quantization and custom input patch
  pruning. Every configuration is benchmarked on accuracy (MASE / WQL) against
  latency and memory.

## Data sources
| Domain | Dataset | Why |
|---|---|---|
| Economics | FRED (Federal Reserve Economic Data) | Low-frequency macro series with revisions, regime changes, and real structural breaks |
| Telecommunications | Telecom Italia Big Data Challenge (Milan) | High-frequency mobile network traffic (SMS, calls, internet) with strong daily and weekly cycles |

## Architecture

```
 FRED API ─────────┐                        ┌──────────────────────────┐
                   ├─▶ Airflow DAGs ──────▶ │ ClickHouse               │
 Harvard Dataverse ┘   ingest ─▶ quality    │  observations            │
 (Telecom Italia)                           │  series_meta             │
                                            │  quality_flags           │
                                            │  benchmark_results       │
                                            └────────────┬─────────────┘
                                                         │
                          Chronos-Bolt (fp32 / int8, patch pruning) benchmark
```

| DAG | Schedule | What it does |
|---|---|---|
| `fred_ingest` | daily 06:00 UTC | Pulls each configured series, re-reading a 400-day revision window, then runs the quality checks |
| `telecom_milan_ingest` | manual | Downloads the daily grid files (Nov 2013 – Jan 2014), aggregates city-wide and per-cell hourly traffic, then runs the quality checks |

`observations` is a `ReplacingMergeTree`, so re-ingesting a window (FRED
revisions, a re-run backfill) replaces rows instead of duplicating them.

## Quickstart

```bash
make env            # creates .env from .env.example; add your FRED_API_KEY
make up             # Airflow on http://localhost:8080 (admin / admin), ClickHouse on :8123
```

Unpause and trigger the DAGs in the Airflow UI. The Telecom Italia files are a
few hundred MB per day; if Dataverse asks for authentication, set
`DATAVERSE_API_TOKEN` in `.env`, or download the `sms-call-internet-mi-*.txt`
files by hand into `data/raw/telecom_milan/`.

To work on the code or run the benchmark on the host:

```bash
make install        # pip install -e ".[dev,forecast]"
make test
make benchmark-milan
make benchmark-fred
```

Sources, series, grid cells and peer groups live in
[configs/sources.yaml](configs/sources.yaml).

## Data quality: data error or drift?

Rule-based checks catch the unambiguous problems first: missing values,
holes in the timestamp grid, duplicate timestamps, negative counts and stuck
feeds ([checks.py](src/daf/quality/checks.py)).

Sudden changes are harder, because a broken feed and a real change in the
world can produce the same jump. The detector
([drift.py](src/daf/quality/drift.py)) removes the seasonal profile, builds a
causal baseline (robust level plus local slope from the previous window) and
scores each point as a robust z-score. Candidate events are large single
surprises plus the onsets of a two-sided CUSUM, which catches moderate but
persistent shifts that never produce a single outlier.

The classifier ([classify.py](src/daf/quality/classify.py)) then looks at what
happens *after* the change:

| Evidence after the change | Interpretation | Verdict |
|---|---|---|
| Values freeze (often at 0) | Stuck sensor or collection outage | `data_error` |
| Level moves by about 10^k | Unit or scaling change | `data_error` |
| Too few points observed yet | Spike and shift still look alike | `pending` |
| Values return to the expected level | Transient spike or bad value | `data_error` |
| Values stay at the new level | Persistent change in the process | `drift` |

Related series act as witnesses. Other grid cells for Milan, or indicators in
the same peer group for FRED (unemployment, payrolls, jobless claims, ...). A
persistent shift that peers share is real drift with higher confidence. One no
peer shares is downgraded to `inconclusive`, since it could be a definition or
reporting change. A spike that peers share is a real short-lived event, not a
bad value. Peers keep their own frequency and seasonal period, so weekly
claims can confirm a monthly unemployment shift.

Every verdict lands in `quality_flags` with a confidence, a human-readable
reason and JSON evidence (peak z, persistence, peer support, ...). Each run
replaces a series' flags, so a `pending` verdict resolves once enough data
arrives.

## Forecasting and compression

[Chronos-Bolt](https://huggingface.co/amazon/chronos-bolt-small) is a T5
encoder-decoder that cuts the context into 16-step patches, one encoder token
each. Two independent levers make it cheaper on CPU:

- **int8 dynamic quantization** ([quantization.py](src/daf/forecasting/quantization.py)).
  Linear layers in the encoder and decoder get int8 weights. The input and
  output patch embeddings stay in float32 because they map raw values in and
  quantiles out.
- **Input patch pruning** ([patch_pruning.py](src/daf/forecasting/patch_pruning.py)).
  Whole patches are dropped from the context before inference, keeping the
  model's own patch boundaries, so the encoder sees fewer tokens. The newest
  patches are always kept. The older ones are chosen by strategy:
  - `recent`: plain truncation (the baseline).
  - `seasonal`: patches at the same point of the cycle as the forecast horizon.
  - `saliency`: patches with the most variation or the sharpest breaks (peaks,
    level shifts).

## Benchmark

`python -m daf.forecasting.benchmark` evaluates every variant on the same
rolling-origin windows and reports:

- **Accuracy**: MASE of the median and WQL over 9 quantiles, using the
  definitions from the Chronos paper.
- **Speed**: p50/p95 latency per batch and throughput.
- **Size**: serialized model size and average encoder tokens.

A seasonal-naive row is included as the floor any model has to beat.
Results go to `reports/benchmark_<source>.md` / `.csv` and, with
`--to-clickhouse`, to the `benchmark_results` table.

Default variants: `fp32`, `int8`, `fp32+{recent,seasonal,saliency}50`,
`int8+{seasonal,saliency}50`, `int8+saliency25` (the number is the share of
patches kept).

## Repository layout

```
airflow/              Airflow image and DAGs
clickhouse/init/      Table definitions (run on first start)
configs/sources.yaml  Series, grid cells, peer groups
src/daf/
  ingest/             FRED and Telecom Italia loaders
  quality/            Validity checks, change detection, error-vs-drift classifier
  forecasting/        Metrics, Chronos-Bolt wrapper, quantization, patch pruning, benchmark
  storage/            ClickHouse access
tests/                Unit tests (no network, no database)
```
