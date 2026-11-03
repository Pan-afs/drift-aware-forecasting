ARG AIRFLOW_VERSION=2.10.3
FROM apache/airflow:${AIRFLOW_VERSION}-python3.11

ARG AIRFLOW_VERSION=2.10.3
COPY airflow/requirements.txt /tmp/requirements.txt

# Pin Airflow itself so extra packages cannot silently up/downgrade it.
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt
