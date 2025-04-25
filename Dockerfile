# Build stage
FROM python:3.11-slim AS builder

# Install uv
RUN apt-get update && apt-get install -y curl
RUN curl -Ls https://astral.sh/uv/install.sh | bash
ENV PATH="/root/.cargo/bin:$PATH"

# Set uv environment variables
ENV UV_LINK_MODE=copy 

WORKDIR /app

# Copy requirements only first (for better caching)
COPY requirements.txt* pyproject.toml* ./

# Install dependencies
RUN uv sync

# Copy the rest of the application
COPY . .

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy application from builder
COPY --from=builder /app /app

# Expose Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["python", "-m", "streamlit", "run", "src/chatbot/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
