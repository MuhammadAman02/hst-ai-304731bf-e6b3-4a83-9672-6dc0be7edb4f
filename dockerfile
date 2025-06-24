# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wheel setuptools && \
    pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    HOST=0.0.0.0 \
    DEBUG=false

# Copy wheels from builder stage
COPY --from=builder /app/wheels /app/wheels

# Install runtime dependencies and Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --no-index --find-links=/app/wheels/ /app/wheels/* \
    && rm -rf /app/wheels

# Create necessary directories
RUN mkdir -p /app/logs /app/data

# Copy application code
COPY app /app/app
COPY main.py requirements.txt /app/

# Copy configuration files
COPY .env.example /app/.env.example
COPY fly.toml /app/fly.toml

# Set proper permissions
RUN chmod +x /app/main.py

# Expose the port the app runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"]