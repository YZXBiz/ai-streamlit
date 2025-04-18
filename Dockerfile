FROM python:3.10-slim

WORKDIR /app

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

# Install dependencies for all packages
RUN uv pip install -e ".[all]"

# Create necessary directories (only those that aren't mounted volumes)
RUN mkdir -p cache

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Create wrapper script for consistent execution (matching Makefile pattern)
RUN echo '#!/bin/bash\nexec uv run "$@"' > /usr/local/bin/run-python && \
    chmod +x /usr/local/bin/run-python

# Command to run when the container starts
ENTRYPOINT ["run-python", "-m", "clustering"]
CMD ["--help"]
