# Multi-stage Dockerfile for ClimateVision
# Builds the React frontend, then packages the FastAPI backend + static files.

# -----------------------------------------------------------------------------
# Stage 1: Build the frontend
# -----------------------------------------------------------------------------
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
ENV VITE_API_BASE_URL=
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2: Python API runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS api

WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

# Install system dependencies required by rasterio, opencv, and other geospatial libs.
# build-essential and python3-dev are needed to compile packages like stringzilla
# that do not provide pre-built wheels for this platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgdal-dev \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install the ClimateVision package
COPY setup.py README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Copy built frontend from the first stage
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Copy config, models, and utility scripts
COPY config.yaml ./
COPY config/ ./config/
COPY models/ ./models/
COPY scripts/ ./scripts/
RUN chmod +x scripts/*.sh

# Create writable directories for SQLite and outputs
RUN mkdir -p /app/outputs /app/data /app/logs

EXPOSE 8000

# Note: GEE credentials and other secrets are supplied at runtime via env vars.
# Do not bake credentials into the image.
# The default CMD uses port 8000; Render overrides this via the PORT env var.
CMD ["./scripts/render_entrypoint.sh"]
