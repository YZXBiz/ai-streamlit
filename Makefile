.PHONY: install update format lint type-check test clean build run-all setup-configs setup-tests run-% help dagster-ui dagster-test dagster-job-% run-dagster-% docs docs-server docs-deps

# Default Python interpreter
PYTHON := uv run

# Package name
PACKAGE_NAME := clustering

# Default directories
SRC_DIR := src
TESTS_DIR := tests
CONFIGS_DIR := configs

# Default environment
ENV_FILE := .env

# Default Dagster environment
DAGSTER_ENV := dev

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
	uv run -m ruff format $(SRC_DIR) $(TESTS_DIR)

# Lint code
lint:
	@echo "Lint code"
	uv run -m ruff check $(SRC_DIR) $(TESTS_DIR) --fix

# Type check
type-check:
	@echo "Type check"
	uv run -m mypy $(SRC_DIR) $(TESTS_DIR)
	uv run -m pyright $(SRC_DIR) $(TESTS_DIR)

# Run tests
test:
	@echo "Run tests"
	uv run -m pytest $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=term --cov-report=xml -v

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
	find . -name ".tmp_dagster*" -type d -exec rm -rf {} +

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

# Start Dagster UI
dagster-ui:
	@echo "Starting Dagster UI with $(DAGSTER_ENV) environment"
	uv run -m $(PACKAGE_NAME).dagster.app --env $(DAGSTER_ENV)

# Run Dagster tests
dagster-test:
	@echo "Running Dagster tests"
	uv run -m pytest $(TESTS_DIR)/test_dagster_implementation.py -v

# Run a specific Dagster job with the dagster CLI
dagster-job-%:
	@echo "Running Dagster job $*"
	uv run -m dagster job execute -f $(PACKAGE_NAME).dagster.definitions:defs $*

# Run a specific Dagster job with our run_dagster.py script
run-dagster-%:
	@echo "Running Dagster job $* in environment $(DAGSTER_ENV)"
	uv run -m $(PACKAGE_NAME).dagster.run_dagster $* --env $(DAGSTER_ENV)

# Install documentation dependencies
docs-deps:
	@echo "Installing documentation dependencies"
	uv add --group docs sphinx sphinx-rtd-theme sphinx-autodoc-typehints sphinx-autobuild

# Documentation
docs: docs-deps
	@echo "Building documentation"
	cd docs && $(MAKE) html

# Start documentation server
docs-server: docs-deps
	@echo "Starting documentation server at http://localhost:8000"
	cd docs && uv run -m sphinx_autobuild source build/html --port 8000 --host 0.0.0.0

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
	@echo "  run-<job>     - Run a specific job using legacy method (e.g., make run-internal_clustering)"
	@echo "  setup-configs - Create configs directory"
	@echo "  setup-tests   - Create test directories and files"
	@echo "  dagster-ui    - Start Dagster UI (set DAGSTER_ENV for environment)"
	@echo "  dagster-test  - Run Dagster tests"
	@echo "  dagster-job-<job> - Run a specific Dagster job using the dagster CLI"
	@echo "  run-dagster-<job>  - Run a specific Dagster job using our run_dagster.py script"
	@echo "  docs          - Build documentation"
	@echo "  docs-server   - Start documentation server" 