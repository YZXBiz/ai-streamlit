FROM python:3.10-slim AS builder

WORKDIR /build

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv directly instead of installing Rust + building uv
RUN curl -fsSL https://astral.sh/uv/install.sh | bash

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./
COPY clustering-pipeline/pyproject.toml clustering-pipeline/
COPY clustering-cli/pyproject.toml clustering-cli/
COPY clustering-shared/pyproject.toml clustering-shared/
COPY clustering-dashboard/pyproject.toml clustering-dashboard/

# Copy the source code
COPY clustering-pipeline/src/ clustering-pipeline/src/
COPY clustering-cli/src/ clustering-cli/src/
COPY clustering-shared/src/ clustering-shared/src/
COPY clustering-dashboard/src/ clustering-dashboard/src/

# Install dependencies and build wheel packages
RUN uv pip wheel --wheel-dir /wheels -e ".[all]"

# Production stage
FROM python:3.10-slim

WORKDIR /app

# Install minimal system dependencies for runtime only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder stage
COPY --from=builder /wheels /wheels

# Install from wheels (faster and more reproducible)
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Create necessary directories for runtime
RUN mkdir -p cache data/internal data/external data/merging logs

# Copy configuration files
COPY configs/ /app/configs/

# Set environment variables
ENV PYTHONPATH=/app
ENV DAGSTER_HOME=/app/dagster_home

# Create wrapper script for consistent execution (matching Makefile pattern)
RUN echo '#!/bin/bash\nexec python -m "$@"' > /usr/local/bin/run-python && \
    chmod +x /usr/local/bin/run-python

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Command to run when the container starts
ENTRYPOINT ["run-python", "clustering"]
CMD ["--help"]
