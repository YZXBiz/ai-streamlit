.PHONY: install dev-install format lint type-check test clean build run-all

# Default Python interpreter
PYTHON := python3

# Package name
PACKAGE_NAME := clustering

# Default directories
SRC_DIR := src
TESTS_DIR := tests
CONFIGS_DIR := configs

# Default environment
ENV_FILE := .env

# Install production dependencies
install:
	@echo "Sync all dependencies"
	uv sync --all-packages 

# Update all dependencies
update:
	@echo "Update all dependencies"
	uv lock --upgrade && uv sync --all-packages

# Format code
format:
	@echo "Format code"
	ruff format $(SRC_DIR) $(TESTS_DIR)

# Lint code
lint:
	@echo "Lint code"
	ruff check $(SRC_DIR) $(TESTS_DIR) --fix

# Type check
type-check:
	@echo "Type check"
	mypy $(SRC_DIR) $(TESTS_DIR)
	pyright $(SRC_DIR) $(TESTS_DIR)

# Run tests
test:
	@echo "Run tests"
	pytest $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=term --cov-report=xml -v

# Clean up build artifacts and cache files
clean:
	@echo "Clean up build artifacts and cache files"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.pyc" -delete
	find . -name "*.log" -type f -delete
	poe clean_mlruns
	poe clean_cache

# Build package	
build:
	@echo "Build package"
	uv build --all-packages

# Create configs directory if it doesn't exist
setup-configs:
	@echo "Create configs directory"
	mkdir -p $(CONFIGS_DIR)

# Setup test environment
setup-tests:
	@echo "Create test directories"
	mkdir -p $(TESTS_DIR)/unit
	mkdir -p $(TESTS_DIR)/integration
	@echo "Create test files"
	touch $(TESTS_DIR)/__init__.py
	touch $(TESTS_DIR)/unit/__init__.py
	touch $(TESTS_DIR)/integration/__init__.py
	touch $(TESTS_DIR)/conftest.py

# Run a specific job
run-%:
	@echo "Run job"
	uv run -m $(PACKAGE_NAME) $(CONFIGS_DIR)/$*.yml

# Help
help:
	@echo "Available targets:"
	@echo "  install       - Install production dependencies"
	@echo "  dev-install   - Install development dependencies"
	@echo "  format        - Format code with ruff"
	@echo "  lint          - Lint code with ruff"
	@echo "  type-check    - Type check with mypy and pyright"
	@echo "  test          - Run tests with pytest"
	@echo "  clean         - Clean build artifacts and cache files"
	@echo "  build         - Build package"
	@echo "  run-all       - Run all clustering pipeline"
	@echo "  run-<job>     - Run a specific job (e.g., make run-internal_clustering)"
	@echo "  setup-configs - Create configs directory"
	@echo "  setup-tests   - Create test directories and files" 