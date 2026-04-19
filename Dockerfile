FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install project in editable mode so climatevision module is available
RUN pip install -e .

EXPOSE 8000

# Set environment variable to ensure python output is not buffered
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

# Run the FastAPI server
CMD ["uvicorn", "climatevision.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
