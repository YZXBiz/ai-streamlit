FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Set uv environment variables
ENV UV_LINK_MODE=copy 

# Copy the entire project
COPY . .

# Install dependencies
RUN uv sync

# Install additional system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Expose Streamlit port
EXPOSE 8501
# Expose FastAPI port
EXPOSE 8000

# Create startup script
RUN echo '#!/bin/bash\n\
    # Start FastAPI backend\n\
    uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 & \n\
    # Start Streamlit frontend\n\
    uv run streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0\n\
    # Wait for any process to exit\n\
    wait -n\n\
    # Exit with status of process that exited first\n\
    exit $?' > /app/start.sh && chmod +x /app/start.sh

# Command to run both services
CMD ["/app/start.sh"]
