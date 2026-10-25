#!/usr/bin/env bash
# Starts OpenObserve and the OpenTelemetry Collector on the observability
# container (stu27_sys2). Needs ./openobserve and ./otelcol next to this file
# and a file `secrets.env` with:
#   ZO_ROOT_USER_EMAIL=...
#   ZO_ROOT_USER_PASSWORD=...
set -euo pipefail
cd "$(dirname "$0")"

set -a; source ./secrets.env; set +a
export OO_BASIC_AUTH=$(printf '%s:%s' "$ZO_ROOT_USER_EMAIL" "$ZO_ROOT_USER_PASSWORD" | base64 -w0)

pkill -x openobserve || true
pkill -x otelcol || true
sleep 1

# OpenObserve: storage + web UI. Port 3000 is published as 10.1.75.51:3306.
export ZO_DATA_DIR="$PWD/data"
export ZO_HTTP_PORT=3000
export ZO_GRPC_PORT=5081
export ZO_TELEMETRY=false              # no usage reporting to openobserve.ai
# Keep memory well inside the 512 MB container. OpenObserve sizes buckets and
# thread pools by CPU count, and inside the container it sees all 120 host
# cores; one bucket and small pools match the single vCPU actually available.
export ZO_MEMORY_CACHE_MAX_SIZE=64
export ZO_MEMORY_CACHE_DATAFUSION_MAX_SIZE=128
export ZO_MEM_TABLE_MAX_SIZE=96
export ZO_MEM_TABLE_BUCKET_NUM=1
export ZO_MAX_FILE_RETENTION_TIME=60   # write in-memory data to disk every minute
export ZO_HTTP_WORKER_NUM=2
export ZO_QUERY_THREAD_NUM=2
export ZO_MEM_DUMP_THREAD_NUM=1
export ZO_FILE_MOVE_THREAD_NUM=1
export ZO_FILE_MERGE_THREAD_NUM=1
export ZO_GRPC_RUNTIME_WORKER_NUM=1
export ZO_JOB_RUNTIME_WORKER_NUM=1
export ZO_WAL_RUNTIME_WORKER_NUM=1
nohup ./openobserve > openobserve.log 2>&1 &

for _ in $(seq 30); do
  curl -sf http://127.0.0.1:3000/healthz > /dev/null && break
  sleep 1
done

# Collector: receives OTLP from the pond app, batches, forwards to OpenObserve.
GOMAXPROCS=1 GOMEMLIMIT=90MiB nohup ./otelcol --config ./otelcol-config.yaml > collector.log 2>&1 &
sleep 2
pgrep -a -x 'openobserve|otelcol'
