#!/bin/bash

# Setup script for store clustering dashboard

set -e  # Exit on error

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up store clustering dashboard...${NC}"

# Create data directories if they don't exist
echo -e "${BLUE}Creating data directories...${NC}"
mkdir -p data/internal
mkdir -p data/external
mkdir -p data/raw

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${BLUE}Installing uv package manager...${NC}"
    curl -fsSL https://astral.sh/uv/install.sh | bash
fi

# Install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
uv sync

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${BLUE}Creating .env file...${NC}"
    cat > .env << EOF
# Dashboard Configuration
DATA_DIR=./data
ENV=dev

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EOF
fi

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}Run 'make dashboard' to start the dashboard.${NC}" 