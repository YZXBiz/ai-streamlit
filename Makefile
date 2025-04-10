# Clustering Project Makefile
# ----------------------------------
# Author: Jackson Yang

# Declare all targets as phony (not files)
.PHONY: install update format lint type-check test clean build setup-configs setup-tests \
        dagster-ui dagster-test dagster-job-% run-internal-% run-external-% run-full run-merging \
        docs docs-server docs-deps help version dev run-job setup full-pipeline

# ===== CONFIGURATION =====
# Package info
PACKAGE_NAME := clustering
VERSION := 1.0.0
AUTHOR := Jackson Yang

# Directories
SRC_DIR := src
TESTS_DIR := tests
CONFIGS_DIR := configs
DAGSTER_HOME_DIR := $(shell pwd)/dagster_home

# Environment settings
DAGSTER_ENV := dev
ENV ?= dev

# Tools
PYTHON := uv run

# ===== SETUP TARGETS =====
# Setup project directory structure
setup:
	mkdir -p $(DAGSTER_HOME_DIR)
	touch $(DAGSTER_HOME_DIR)/dagster.yaml

# Install uv and sync dependencies
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

# ===== DEVELOPMENT =====
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

# Start development server
dev: setup
	DAGSTER_HOME=$(DAGSTER_HOME_DIR) python -m dagster dev

# Run tests
test:
	@echo "Run tests"
	uv run --no-deps -m pytest $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=term --cov-report=xml -v

# Show version and author information
version:
	@echo "=========================================="
	@echo "  $(PACKAGE_NAME) version $(VERSION)"
	@echo "  Developed by $(AUTHOR)"
	@echo "  Copyright (c) 2025-2026 $(AUTHOR)"
	@echo "=========================================="

# ===== BUILD & CLEAN =====
# Build package	
build:
	@echo "Build package"
	uv build --all-packages

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
	rm -rf $(DAGSTER_HOME_DIR)/.tmp_*

# ===== DAGSTER COMMANDS =====
# Start Dagster UI
dagster-ui: setup
	@echo "Starting Dagster UI with $(DAGSTER_ENV) environment"
	DAGSTER_HOME=$(DAGSTER_HOME_DIR) uv run -m $(PACKAGE_NAME).dagster.app --env $(DAGSTER_ENV)

# Run Dagster tests
dagster-test: setup
	@echo "Running Dagster tests"
	DAGSTER_HOME=$(DAGSTER_HOME_DIR) uv run -m pytest $(TESTS_DIR)/test_dagster_implementation.py -v

# Run a specific Dagster job with the dagster CLI
dagster-job-%: setup
	@echo "Running Dagster job $*"
	DAGSTER_HOME=$(DAGSTER_HOME_DIR) uv run -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j $*

# Run internal jobs (preprocessing or clustering)
run-internal-%: setup
	@echo "Running internal $* job"
	DAGSTER_HOME=$(DAGSTER_HOME_DIR) uv run -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j internal_$*_job

# Run external jobs (preprocessing or clustering) 
run-external-%: setup
	@echo "Running external $* job"
	DAGSTER_HOME=$(DAGSTER_HOME_DIR) uv run -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j external_$*_job

# Run full pipeline job
run-full: setup
	@echo "Running full pipeline job"
	DAGSTER_HOME=$(DAGSTER_HOME_DIR) uv run -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j full_pipeline_job

# Run merging job
run-merging: setup
	@echo "Running merging job"
	DAGSTER_HOME=$(DAGSTER_HOME_DIR) uv run -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j merging_job

# Run a specific Dagster job
run-job: setup
	DAGSTER_HOME=$(DAGSTER_HOME_DIR) python -m clustering run $(JOB) --env $(ENV)

# Run the full pipeline
full-pipeline: setup
	DAGSTER_HOME=$(DAGSTER_HOME_DIR) python -m clustering run full_pipeline_job --env $(ENV)

# ===== DOCUMENTATION =====
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

# ===== HELP =====
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
	@echo "  dev           - Start Dagster development server"
	@echo "  format        - Format code with ruff"
	@echo "  lint          - Lint code with ruff"
	@echo "  type-check    - Type check with mypy and pyright"
	@echo "  test          - Run tests with pytest"
	@echo "  version       - Display version and author information"
	@echo ""
	@echo "Build & Clean Targets:"
	@echo "  build         - Build package"
	@echo "  clean         - Clean build artifacts and cache files"
	@echo ""
	@echo "Dagster Targets:"
	@echo "  dagster-ui    - Start Dagster UI (set DAGSTER_ENV for environment)"
	@echo "  dagster-test  - Run Dagster tests"
	@echo "  dagster-job-<job>    - Run a specific Dagster job using the dagster CLI"
	@echo "  run-internal-<type>  - Run internal job (preprocessing or clustering)"
	@echo "  run-external-<type>  - Run external job (preprocessing or clustering)"
	@echo "  run-full      - Run the full pipeline job"
	@echo "  run-merging   - Run the merging job for combining internal and external clusters"
	@echo "  run-job       - Run a specific job (usage: make run-job JOB=job_name)"
	@echo "  full-pipeline - Run the full pipeline job"
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