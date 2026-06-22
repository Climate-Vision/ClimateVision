#!/bin/bash
# Local deployment helper for Fly.io.
# Usage: ./scripts/deploy.sh

set -e

cd "$(dirname "$0")/.."

echo "Running test suite..."
.venv/bin/pytest tests/ -q

echo "Building frontend..."
cd frontend
npm ci
VITE_API_BASE_URL= npm run build
cd ..

echo "Deploying to Fly.io..."
fly deploy --remote-only

echo "Deployment complete. Check status with: fly status"
