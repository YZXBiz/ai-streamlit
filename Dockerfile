FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    PATH="/root/.cargo/bin:${PATH}" \
    LOGS_DIR=/app/logs \
    LOG_LEVEL=INFO

# Working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -sSf https://astral.sh/uv/install.sh | sh

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock Makefile ./

# Install dependencies
RUN uv sync

# Copy application code
COPY app ./app
COPY dashboard ./dashboard
COPY tests ./tests
COPY config ./config
COPY docs ./docs

# Create required directories
RUN mkdir -p data/internal data/external data/raw logs

# Set permissions
RUN chmod +x /app/app/main.py

# Volume configuration for persistent data and logs
VOLUME ["/app/data", "/app/logs"]

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501 || exit 1

# Command to run the application
CMD ["make", "dashboard"]
