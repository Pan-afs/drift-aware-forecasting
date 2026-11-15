# drift-aware-forecasting
End-to-end time-series pipeline: Airflow + ClickHouse in Docker Compose load open data (FRED economic indicators, Telecom Italia Milan mobile traffic), automatically tell data errors apart from real distribution drift, and forecast with a compressed Chronos-Bolt model (int8 + patch pruning), reporting accuracy vs. speed.
