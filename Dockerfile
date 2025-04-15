FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for package management
RUN curl -fsSL https://pkg.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo install --force uv

# Copy just the requirements first to leverage Docker caching
COPY pyproject.toml .

# Install dependencies
RUN uv pip install -e .

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p logs configs outputs reports cache

# Set environment variables
ENV PYTHONPATH=/app

# Command to run when the container starts
ENTRYPOINT ["uv", "run", "-m", "clustering"]
CMD ["--help"]
