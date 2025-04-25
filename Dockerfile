FROM ghcr.io/astral-sh/uv:latest

WORKDIR /app

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
