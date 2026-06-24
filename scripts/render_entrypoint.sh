#!/bin/bash
# Render.com entrypoint.
# Writes the GEE service-account key from the environment into a file,
# then starts the ClimateVision API.

set -e

mkdir -p /app/secrets

# Render stores file contents in env vars because it does not support secret
# files on the free tier. Write the JSON key to disk so the app can read it.
if [ -n "$GEE_SERVICE_ACCOUNT_KEY_JSON" ]; then
    echo "$GEE_SERVICE_ACCOUNT_KEY_JSON" > /app/secrets/gee-key.json
fi

# Create writable directories for SQLite and outputs
mkdir -p /app/outputs /app/data /app/logs

export PORT="${PORT:-8000}"

echo "Starting ClimateVision API on port $PORT"
exec uvicorn climatevision.api.main:app --host 0.0.0.0 --port "$PORT" --workers 1
