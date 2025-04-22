.PHONY: help install dev lint format test clean run

# Directory Structure
PACKAGE_DIR = assortment_chatbot
TEST_DIR = tests
BUILD_DIR = build

# Python Settings
PYTHON = python3
VENV = .venv
VENV_BIN = $(VENV)/bin
UV = uv
PIP = $(UV) pip

# App Configuration
PORT ?= 8501

# Help Target - Show available commands
help:
	@echo "Available commands:"
	@echo "  help     : Show this help message"
	@echo "  install  : Set up virtual environment and install dependencies"
	@echo "  dev      : Install development dependencies"
	@echo "  lint     : Run linting checks"
	@echo "  format   : Format code with ruff"
	@echo "  test     : Run tests"
	@echo "  clean    : Remove build artifacts and cache files"
	@echo "  run      : Run the Streamlit application"

# Setup virtual environment and install dependencies
install:
	@echo "==> Creating virtual environment and installing dependencies"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -e .

# Install development dependencies
dev: install
	@echo "==> Installing development dependencies"
	$(PIP) install ruff mypy pytest

# Run linting checks
lint:
	@echo "==> Running linting and type checks"
	$(UV) run -m ruff check $(PACKAGE_DIR)
	$(UV) run -m mypy $(PACKAGE_DIR)

# Format code
format:
	@echo "==> Formatting code with ruff"
	$(UV) run -m ruff format $(PACKAGE_DIR) $(TEST_DIR)
	@echo "✓ Code formatting complete"

# Run tests
test:
	@echo "==> Running tests"
	$(UV) run -m pytest $(TEST_DIR)

# Clean build artifacts and cache files
clean:
	@echo "==> Cleaning build artifacts and cache files"
	rm -rf $(BUILD_DIR) *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	@echo "✓ Cleaned build artifacts and cache files"

# Run the application
run:
	@echo "==> Starting Streamlit app on port $(PORT)"
	$(UV) run main.py 