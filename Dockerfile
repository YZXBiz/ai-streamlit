FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN apt-get update && apt-get install -y curl
RUN curl -Ls https://astral.sh/uv/install.sh | bash
ENV PATH="/root/.cargo/bin:$PATH"

# Set uv environment variables
ENV UV_LINK_MODE=copy 

# Copy the entire project
COPY . .

# Install everything in one go
RUN uv sync

# Expose Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["uv", "run", "streamlit", "run", "src/chatbot/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
