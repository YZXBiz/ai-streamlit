################################################################################
# Store Clustering - Data Pipeline Project Makefile
################################################################################
# 
# Author: Jackson Yang <Jackson.Yang@cvshealth.com>
# Version: 1.0.0
# Copyright (c) 2025 CVS Health
#
# Description:
#   This Makefile provides targets for developing, testing, and running the 
#   Store Clustering data pipeline project built with Dagster.
#
################################################################################

################################################################################
# CONFIGURATION
################################################################################

# Package information
PACKAGE_NAME := clustering
VERSION := 1.0.0
AUTHOR := Jackson Yang

# Directory paths
SRC_DIR := src 
TESTS_DIR := tests
DAGSTER_CONFIG_DIR := $(SRC_DIR)/$(PACKAGE_NAME)/dagster/resources/configs
DOCS_DIR := docs 

# Include .env file if it exists
# This loads environment variables from .env before setting defaults
-include .env

# Environment variables (defaults if not set in environment or .env)
# Example: make dev DATA_DIR=/custom/path
DAGSTER_HOME_DIR ?= $(shell pwd)/dagster_home
DATA_DIR ?= $(shell pwd)/data
INTERNAL_DATA_DIR ?= $(DATA_DIR)/internal
EXTERNAL_DATA_DIR ?= $(DATA_DIR)/external
MERGING_DATA_DIR ?= $(DATA_DIR)/merging
LOGS_DIR ?= $(shell pwd)/logs

# Environment settings
DAGSTER_ENV ?= dev
ENV ?= dev

# Set UV_LINK_MODE if not already defined, default to copy
UV_LINK_MODE ?= copy

# Export environment variables for child processes
export DAGSTER_HOME = $(DAGSTER_HOME_DIR)
export DATA_DIR
export INTERNAL_DATA_DIR
export EXTERNAL_DATA_DIR
export MERGING_DATA_DIR
export UV_LINK_MODE

# Python command
PYTHON := uv run

################################################################################
# HELP TARGET
################################################################################

.PHONY: help
help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\n\033[1mStore Clustering Makefile Help\033[0m\n"} \
		/^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	@echo ""
	@echo "Usage examples:"
	@echo "  make install             # Install dependencies"
	@echo "  make dev                 # Start Dagster dev server"
	@echo "  make run-full ENV=prod   # Run full pipeline in production environment"
	@echo "  make dagster-job-JOB     # Run a specific Dagster job"
	@echo ""
	@echo "Environment variables priority:"
	@echo "  1. Command line (make dev DATA_DIR=/custom/path)"
	@echo "  2. Shell environment variables"
	@echo "  3. Variables from .env file"
	@echo "  4. Default values in Makefile"
	@echo ""
	@echo "Configuration:"
	@echo "  Dagster configs: $(DAGSTER_CONFIG_DIR)"
	@echo ""
	@echo "Project info:"
	@echo "  Version: $(VERSION)"
	@echo "  Author:  $(AUTHOR)"
	@echo ""

# Default target
.DEFAULT_GOAL := help

################################################################################
# SETUP TARGETS
################################################################################

##@ Setup

.PHONY: print-env
print-env: ## Display current environment variable values
	@echo "==> Current environment configuration:"
	@echo "- DAGSTER_HOME: $(DAGSTER_HOME_DIR)"
	@echo "- DATA_DIR: $(DATA_DIR)"
	@echo "- INTERNAL_DATA_DIR: $(INTERNAL_DATA_DIR)"
	@echo "- EXTERNAL_DATA_DIR: $(EXTERNAL_DATA_DIR)"
	@echo "- MERGING_DATA_DIR: $(MERGING_DATA_DIR)"
	@echo "- LOGS_DIR: $(LOGS_DIR)"
	@echo "- DAGSTER_ENV: $(DAGSTER_ENV)"
	@echo "- ENV: $(ENV)"

################################################################################
# DEPENDENCY MANAGEMENT
################################################################################

##@ Dependencies

.PHONY: install update docs-deps
install: ## Install uv package manager and project dependencies
	@echo "==> Checking if uv is installed..."
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Installing uv package manager..."; \
		curl -Ls https://astral.sh/uv/install.sh | bash; \
		echo "✓ uv installed successfully"; \
	else \
		echo "✓ uv is already installed"; \
	fi
	@echo "==> Installing project dependencies"
	@uv sync --all-packages
	@echo "✓ All dependencies installed successfully"

update: ## Update all project dependencies to latest versions
	@echo "==> Updating all dependencies"
	@uv lock --upgrade && uv sync --all-packages --no-install-package kaleido
	@echo "✓ Dependencies updated successfully"

docs-deps: ## Install documentation dependencies
	@echo "==> Installing documentation dependencies"
	@uv add --group docs sphinx sphinx-rtd-theme sphinx-autodoc-typehints sphinx-autobuild
	@echo "✓ Documentation dependencies installed"

################################################################################
# DEVELOPMENT TARGETS
################################################################################

##@ Development

.PHONY: dev format lint type-check check-all version
dev: ## Start Dagster development server without creating directories
	@echo "==> Starting Dagster development server"
	@$(PYTHON) -m dagster dev -m $(PACKAGE_NAME).dagster.definitions
	@echo "✓ Dagster development server stopped"

format: ## Format code with ruff formatter
	@echo "==> Formatting code with ruff"
	@$(PYTHON) -m ruff format $(SRC_DIR) $(TESTS_DIR)
	@echo "✓ Code formatting complete"

lint: ## Lint code and auto-fix issues where possible
	@echo "==> Linting code with ruff"
	@$(PYTHON) -m ruff check $(SRC_DIR) $(TESTS_DIR) --fix
	@echo "✓ Code linting complete"

type-check: ## Run type checking with mypy and pyright
	@echo "==> Running mypy type checker"
	@$(PYTHON) -m mypy $(SRC_DIR) $(TESTS_DIR)
	@echo "==> Running pyright type checker"
	@$(PYTHON) -m pyright $(SRC_DIR) $(TESTS_DIR)
	@echo "✓ Type checking complete"

check-all: format lint type-check ## Run all code quality checks
	@echo "✓ All code quality checks completed successfully"

version: ## Display version and author information
	@echo "=================================================="
	@echo "  $(PACKAGE_NAME) version $(VERSION)"
	@echo "  Developed by $(AUTHOR)"
	@echo "  Copyright (c) 2025 CVS Health"
	@echo "=================================================="

################################################################################
# TESTING TARGETS
################################################################################

##@ Testing

.PHONY: test test-unit test-integration dagster-test
test: ## Run tests with coverage reporting
	@echo "==> Running tests with coverage"
	@$(PYTHON) --no-deps -m pytest $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=term --cov-report=xml -v
	@echo "✓ Tests completed"

test-unit: ## Run only unit tests
	@echo "==> Running unit tests"
	@$(PYTHON) --no-deps -m pytest $(TESTS_DIR)/unit -v
	@echo "✓ Unit tests completed"

test-integration: ## Run only integration tests
	@echo "==> Running integration tests"
	@$(PYTHON) --no-deps -m pytest $(TESTS_DIR)/integration -v
	@echo "✓ Integration tests completed"

dagster-test: ## Run Dagster-specific tests
	@echo "==> Running Dagster implementation tests"
	@$(PYTHON) -m pytest $(TESTS_DIR)/test_dagster_implementation.py -v
	@echo "✓ Dagster tests completed"

################################################################################
# BUILD & CLEAN TARGETS
################################################################################

##@ Build and Deploy

.PHONY: build clean
build: ## Build the Python package for distribution
	@echo "==> Building package for distribution"
	@uv build --all-packages
	@echo "✓ Build complete. Files available in the dist/ directory"

clean: ## Clean up all build artifacts and temporary files
	@echo "==> Cleaning build artifacts and cache files"
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf .coverage
	@rm -rf coverage.xml
	@rm -rf .pytest_cache/
	@rm -rf .ruff_cache/
	@rm -rf .mypy_cache/
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name "*.pyc" -delete
	@find . -name "*.log" -type f -delete
	@find . -name ".tmp_dagster*" -type d -exec rm -rf {} +
	@rm -rf $(DAGSTER_HOME_DIR)/.tmp_*
	@echo "✓ Clean complete"

clean-temp: ## Clean only temporary dagster files
	@echo "==> Cleaning temporary Dagster files"
	@find . -name ".tmp_dagster*" -type d -exec rm -rf {} +
	@echo "✓ Temporary files cleaned"

################################################################################
# DAGSTER COMMANDS
################################################################################

##@ Dagster Commands

.PHONY: dagster-ui dagster-job-%
dagster-ui: ## Start the Dagster UI web interface
	@echo "==> Starting Dagster UI with $(DAGSTER_ENV) environment"
	@$(PYTHON) -m $(PACKAGE_NAME).dagster.app --env $(DAGSTER_ENV)
	@echo "✓ Dagster UI started. Access at http://localhost:3000"

dagster-job-%: ## Run a specific Dagster job (usage: make dagster-job-JOB_NAME)
	@echo "==> Running Dagster job: $*"
	@$(PYTHON) -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j $*
	@echo "✓ Job $* completed"

################################################################################
# PIPELINE WORKFLOWS
################################################################################

##@ Pipeline Workflows

# Direct Dagster CLI commands
.PHONY: run-internal-% run-external-% run-merging run-full run-job full-pipeline
run-internal-%: ## Run internal pipeline jobs (usage: make run-internal-preprocessing OR run-internal-ml)
	@echo "==> Running internal $* pipeline"
	@$(PYTHON) -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j internal_$*_job
	@echo "✓ Internal $* pipeline completed"

run-external-%: ## Run external pipeline jobs (usage: make run-external-preprocessing OR run-external-ml)
	@echo "==> Running external $* pipeline"
	@$(PYTHON) -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j external_$*_job
	@echo "✓ External $* pipeline completed"

run-merging: ## Run only the cluster merging job
	@echo "==> Running merging job"
	@$(PYTHON) -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j merging_job
	@echo "✓ Merging job completed"

run-full: ## Run the complete pipeline (internal, external, and merging)
	@echo "==> Running full pipeline job"
	@$(PYTHON) -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j full_pipeline_job
	@echo "✓ Full pipeline completed"

# Custom CLI commands
run-job: ## Run a specific job with environment (usage: make run-job JOB=job_name ENV=prod)
	@echo "==> Running job $(JOB) in $(ENV) environment"
	@if [ -z "$(JOB)" ]; then \
		echo "Error: JOB parameter is required. Usage: make run-job JOB=job_name ENV=env_name"; \
		exit 1; \
	fi
	@$(PYTHON) -m $(PACKAGE_NAME) run $(JOB) --env $(ENV)
	@echo "✓ Job $(JOB) completed"

full-pipeline: ## Run the full pipeline with specified environment (usage: make full-pipeline ENV=prod)
	@echo "==> Running full pipeline in $(ENV) environment"
	@$(PYTHON) -m $(PACKAGE_NAME) run full_pipeline_job --env $(ENV)
	@echo "✓ Full pipeline completed"

################################################################################
# MEMORY MANAGEMENT
################################################################################

##@ Memory Management

.PHONY: run-memory-optimized run-visualization
run-memory-optimized: ## Run a job with memory optimization settings (usage: make run-memory-optimized JOB=job_name)
	@echo "==> Running job with memory optimization"
	@if [ -z "$(JOB)" ]; then \
		echo "Error: JOB parameter is required. Usage: make run-memory-optimized JOB=job_name"; \
		exit 1; \
	fi
	@DAGSTER_MULTIPROCESS_MEMORY_OPTIMIZED=1 \
	 $(PYTHON) -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j $(JOB)
	@echo "✓ Memory-optimized job $(JOB) completed"

run-visualization: ## Run visualization job with memory optimization
	@echo "==> Running visualization job with memory optimization"
	@DAGSTER_MULTIPROCESS_MEMORY_OPTIMIZED=1 \
	 DAGSTER_MULTIPROCESS_CHUNK_SIZE=1 \
	 $(PYTHON) -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j internal_visualization
	@echo "✓ Visualization job completed"

################################################################################
# DOCUMENTATION
################################################################################

##@ Documentation

.PHONY: docs docs-server
docs: docs-deps ## Build documentation
	@echo "==> Building documentation"
	@cd $(DOCS_DIR) && LC_ALL=C.UTF-8 LANG=C.UTF-8 $(MAKE) html
	@echo "✓ Documentation built successfully. Open docs/build/html/index.html to view"

docs-server: docs-deps ## Start documentation server (http://localhost:8000)
	@echo "==> Starting documentation server"
	@cd $(DOCS_DIR) && LC_ALL=C.UTF-8 LANG=C.UTF-8 $(PYTHON) -m sphinx_autobuild source build/html --port 8000 --host 0.0.0.0
	@echo "✓ Documentation server running at http://localhost:8000" 