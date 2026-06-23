#!/bin/bash
# Run the ClimateVision API server
# Usage: ./run_api.sh [port]

set -e
cd "$(dirname "$0")"

PORT="${1:-8000}"

if [ ! -d ".venv" ]; then
  echo "Virtual environment not found. Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e ."
  exit 1
fi

source .venv/bin/activate

# Avoid OpenMP sandbox issues; fix NumPy/PyTorch compatibility in spawned workers
export OMP_NUM_THREADS=1

echo "Starting ClimateVision API on http://0.0.0.0:$PORT"
echo "  Health:     http://0.0.0.0:$PORT/api/health"
echo "  API docs:  http://0.0.0.0:$PORT/docs"
echo ""
exec uvicorn climatevision.api.main:app --host 0.0.0.0 --port "$PORT" --workers 1
