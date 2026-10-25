#!/usr/bin/env bash
# (Re)starts the pond app on the app container (stu27_sys1, ~/pond) with
# OpenTelemetry exporting to the collector on stu27_sys2.
set -euo pipefail
cd "$(dirname "$0")"

# Only this app's server; ProximitySearchAPI on port 5000 is left alone.
pkill -f 'uvicorn app.main:app --app-dir backend' || true
sleep 1

export OTEL_SERVICE_NAME=pond-planner
export OTEL_EXPORTER_OTLP_ENDPOINT=http://172.17.0.107:4318   # collector, docker network
export OTEL_METRIC_EXPORT_INTERVAL=15000
export POND_ENVIRONMENT=production

nohup venv/bin/uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 3000 \
  --workers 1 > server.log 2>&1 &

for _ in $(seq 30); do
  curl -sf http://127.0.0.1:3000/api/health > /dev/null && break
  sleep 1
done
curl -s http://127.0.0.1:3000/api/health; echo
