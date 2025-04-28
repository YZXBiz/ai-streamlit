#!/bin/bash

# A script to manually run the backend tests by directly using pytest
# This bypasses the Makefile and configures the environment directly

set -e

# Set up environment
export PYTHONPATH=/workspaces/chatbot-assortment
export ENV=test

# Create any needed directories
mkdir -p backend/logs backend/data/uploads backend/data/vector_store

# Create test environment file if it doesn't exist
if [ ! -f backend/.env.test ]; then
  cat > backend/.env.test << EOF
# API configuration
OPENAI_API_KEY=sk-dummy-key-for-testing

# Database configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres 
DB_PASSWORD=postgres
DB_NAME=chatbot_test
DB_ECHO=False

# DuckDB configuration
DUCKDB_PATH=:memory:

# Storage configuration
DATA_DIR=./data
LOGS_DIR=./logs
STORAGE_PATH=./data/uploads
VECTOR_STORE_DIR=./data/vector_store

# Authentication
SECRET_KEY=test-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# CORS settings
CORS_ORIGINS=["http://localhost:8503"]

# API configuration
API_V1_STR=/api/v1

# PandasAI configuration
MEMORY_SIZE=5
ENFORCE_PRIVACY=True
ENABLE_CACHE=True
MAX_RETRIES=2

# Logging
LOG_LEVEL=INFO
EOF
  echo "Created test environment file at backend/.env.test"
fi

# Use the test environment 
cp backend/.env.test backend/.env

echo "==> Running unit tests"
python -m pytest backend/tests/services/test_analyzer_service.py -v

echo "==> Running server tests"
python -m pytest backend/tests/test_server.py -v

echo "==> Testing if app can be imported (smoke test)"
python -c "from backend.app.main import app; print('App imported successfully')"

echo "All tests completed!" 