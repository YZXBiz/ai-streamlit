# Clustering project Makefile
# ----------------------------------
# Author: Jackson Yang

# Declare all targets as phony (not files)
.PHONY: install update format lint type-check test clean build setup-configs setup-tests \
        dagster-ui dagster-test dagster-job-% run-internal-% run-external-% run-full run-merging \
        docs docs-server docs-deps help version

# ===== Configuration =====
# Default Python package manager
PYTHON := uv run

# Package name
PACKAGE_NAME := clustering

# Directory structure
SRC_DIR := src
TESTS_DIR := tests
CONFIGS_DIR := configs

# Default Dagster environment
DAGSTER_ENV := dev

# Author info
AUTHOR := Jackson Yang
VERSION := 1.0.0

# ===== Dependencies & Setup =====
# Install production dependencies
install:
	@echo "Checking if uv is installed..."
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Installing uv..."; \
		curl -Ls https://astral.sh/uv/install.sh | bash; \
	else \
		echo "uv is already installed."; \
	fi
	@echo "Sync all dependencies"
	uv sync --all-packages

# Update all dependencies
update:
	@echo "Update all dependencies"
	uv lock --upgrade && uv sync --all-packages

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

# ===== Development =====
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

# Show version and author information
version:
	@echo "=========================================="
	@echo "  $(PACKAGE_NAME) version $(VERSION)"
	@echo "  Developed by $(AUTHOR)"
	@echo "  Copyright (c) 2025-2026 $(AUTHOR)"
	@echo "=========================================="

# ===== Dagster Commands =====
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
	uv run -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j $*

# Run internal jobs (preprocessing or clustering)
run-internal-%:
	@echo "Running internal $* job"
	uv run -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j internal_$*_job

# Run external jobs (preprocessing or clustering) 
run-external-%:
	@echo "Running external $* job"
	uv run -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j external_$*_job

# Run full pipeline job
run-full:
	@echo "Running full pipeline job"
	uv run -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j full_pipeline_job

# Run merging job
run-merging:
	@echo "Running merging job"
	uv run -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j merging_job

# ===== Documentation =====
# Install documentation dependencies
docs-deps:
	@echo "Installing documentation dependencies"
	uv add --group docs sphinx sphinx-rtd-theme sphinx-autodoc-typehints sphinx-autobuild

# Build documentation
docs: docs-deps
	@echo "Building documentation"
	cd docs && $(MAKE) html

# Start documentation server
docs-server: docs-deps
	@echo "Starting documentation server at http://localhost:8000"
	cd docs && uv run -m sphinx_autobuild source build/html --port 8000 --host 0.0.0.0

# ===== Help =====
help:
	@echo "===== Clustering Project Makefile Help ====="
	@echo "Developed by $(AUTHOR)"
	@echo "Version $(VERSION)"
	@echo ""
	@echo "Setup Targets:"
	@echo "  install       - Install production dependencies"
	@echo "  update        - Update all dependencies"
	@echo "  setup-configs - Create configs directory"
	@echo "  setup-tests   - Create test directories and files"
	@echo ""
	@echo "Development Targets:"
	@echo "  format        - Format code with ruff"
	@echo "  lint          - Lint code with ruff"
	@echo "  type-check    - Type check with mypy and pyright"
	@echo "  test          - Run tests with pytest"
	@echo "  clean         - Clean build artifacts and cache files"
	@echo "  build         - Build package"
	@echo "  version       - Display version and author information"
	@echo ""
	@echo "Dagster Targets:"
	@echo "  dagster-ui    - Start Dagster UI (set DAGSTER_ENV for environment)"
	@echo "  dagster-test  - Run Dagster tests"
	@echo "  dagster-job-<job> - Run a specific Dagster job using the dagster CLI"
	@echo "  run-internal-<type> - Run internal job (preprocessing or clustering)"
	@echo "  run-external-<type> - Run external job (preprocessing or clustering)"
	@echo "  run-full      - Run the full pipeline job"
	@echo "  run-merging   - Run the merging job for combining internal and external clusters"
	@echo ""
	@echo "Documentation Targets:"
	@echo "  docs          - Build documentation"
	@echo "  docs-server   - Start documentation server"
	@echo ""
	@echo "Examples:"
	@echo "  make run-internal-preprocessing  # Run internal preprocessing job"
	@echo "  make run-external-clustering     # Run external clustering job"
	@echo "  make run-merging                 # Run cluster merging job"
	@echo ""
	@echo "Copyright (c) 2023-2024 $(AUTHOR)" 