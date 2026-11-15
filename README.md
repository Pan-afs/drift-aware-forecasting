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

## Status
🚧 Work in progress. Sections below are filled in as components land.
