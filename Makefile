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
CONFIGS_DIR := configs
DAGSTER_HOME_DIR := $(shell pwd)/dagster_home
DOCS_DIR := docs

# Environment settings (can be overridden via command line)
DAGSTER_ENV := dev
ENV ?= dev

# Export UV link mode to fix hardlinking warnings
export UV_LINK_MODE := copy

# Command aliases for Python tools
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

.PHONY: setup
setup: ## Initialize the project directory structure
	@echo "==> Setting up project directory structure"
	@mkdir -p $(DAGSTER_HOME_DIR)
	@touch $(DAGSTER_HOME_DIR)/dagster.yaml
	@echo "✓ Dagster home directory created at $(DAGSTER_HOME_DIR)"

.PHONY: install
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

.PHONY: update
update: ## Update all project dependencies to latest versions
	@echo "==> Updating all dependencies"
	@uv lock --upgrade && uv sync --all-packages --no-install-package kaleido
	@echo "✓ Dependencies updated successfully"

.PHONY: setup-configs
setup-configs: ## Create configuration directories
	@echo "==> Creating configs directory"
	@mkdir -p $(CONFIGS_DIR)
	@echo "✓ Configuration directory created at $(CONFIGS_DIR)"

.PHONY: setup-tests
setup-tests: ## Set up test directory structure
	@echo "==> Creating test directory structure"
	@mkdir -p $(TESTS_DIR)/unit
	@mkdir -p $(TESTS_DIR)/integration
	@echo "==> Creating test initialization files"
	@touch $(TESTS_DIR)/__init__.py
	@touch $(TESTS_DIR)/unit/__init__.py
	@touch $(TESTS_DIR)/integration/__init__.py
	@touch $(TESTS_DIR)/conftest.py
	@echo "✓ Test directory structure created at $(TESTS_DIR)"

################################################################################
# DEVELOPMENT TARGETS
################################################################################

##@ Development

.PHONY: dev
dev: setup ## Start Dagster development server
	@echo "==> Starting Dagster development server"
	@DAGSTER_HOME=$(DAGSTER_HOME_DIR) $(PYTHON) -m dagster dev

.PHONY: format
format: ## Format code with ruff formatter
	@echo "==> Formatting code with ruff"
	@$(PYTHON) -m ruff format $(SRC_DIR) $(TESTS_DIR)
	@echo "✓ Code formatting complete"

.PHONY: lint
lint: ## Lint code and auto-fix issues where possible
	@echo "==> Linting code with ruff"
	@$(PYTHON) -m ruff check $(SRC_DIR) $(TESTS_DIR) --fix
	@echo "✓ Code linting complete"

.PHONY: type-check
type-check: ## Run type checking with mypy and pyright
	@echo "==> Running mypy type checker"
	@$(PYTHON) -m mypy $(SRC_DIR) $(TESTS_DIR)
	@echo "==> Running pyright type checker"
	@$(PYTHON) -m pyright $(SRC_DIR) $(TESTS_DIR)
	@echo "✓ Type checking complete"

.PHONY: check-all
check-all: format lint type-check ## Run all code quality checks (format, lint, type-check)
	@echo "✓ All code quality checks completed successfully"

.PHONY: version
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

.PHONY: test
test: ## Run tests with coverage reporting
	@echo "==> Running tests with coverage"
	@$(PYTHON) --no-deps -m pytest $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=term --cov-report=xml -v
	@echo "✓ Tests completed"

.PHONY: dagster-test
dagster-test: setup ## Run Dagster-specific tests
	@echo "==> Running Dagster implementation tests"
	@DAGSTER_HOME=$(DAGSTER_HOME_DIR) $(PYTHON) -m pytest $(TESTS_DIR)/test_dagster_implementation.py -v
	@echo "✓ Dagster tests completed"

################################################################################
# BUILD & CLEAN TARGETS
################################################################################

##@ Build and Deploy

.PHONY: build
build: ## Build the Python package for distribution
	@echo "==> Building package for distribution"
	@uv build --all-packages
	@echo "✓ Build complete. Files available in the dist/ directory"

.PHONY: clean
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

################################################################################
# DAGSTER WORKFLOW TARGETS
################################################################################

##@ Dagster Commands

.PHONY: dagster-ui
dagster-ui: setup ## Start the Dagster UI web interface
	@echo "==> Starting Dagster UI with $(DAGSTER_ENV) environment"
	@DAGSTER_HOME=$(DAGSTER_HOME_DIR) $(PYTHON) -m $(PACKAGE_NAME).dagster.app --env $(DAGSTER_ENV)
	@echo "✓ Dagster UI started. Access at http://localhost:3000"

.PHONY: dagster-job-%
dagster-job-%: setup ## Run a specific Dagster job (usage: make dagster-job-JOB_NAME)
	@echo "==> Running Dagster job: $*"
	@DAGSTER_HOME=$(DAGSTER_HOME_DIR) $(PYTHON) -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j $*
	@echo "✓ Job $* completed"

################################################################################
# PIPELINE WORKFLOW TARGETS
################################################################################

##@ Pipeline Workflows

.PHONY: run-internal-%
run-internal-%: setup ## Run internal pipeline jobs (usage: make run-internal-preprocessing OR run-internal-ml)
	@echo "==> Running internal $* pipeline"
	@DAGSTER_HOME=$(DAGSTER_HOME_DIR) $(PYTHON) -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j internal_$*_job
	@echo "✓ Internal $* pipeline completed"

.PHONY: run-external-%
run-external-%: setup ## Run external pipeline jobs (usage: make run-external-preprocessing OR run-external-ml)
	@echo "==> Running external $* pipeline"
	@DAGSTER_HOME=$(DAGSTER_HOME_DIR) $(PYTHON) -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j external_$*_job
	@echo "✓ External $* pipeline completed"

.PHONY: run-full
run-full: setup ## Run the complete pipeline (internal, external, and merging)
	@echo "==> Running full pipeline job"
	@DAGSTER_HOME=$(DAGSTER_HOME_DIR) $(PYTHON) -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j full_pipeline_job
	@echo "✓ Full pipeline completed"

.PHONY: run-merging
run-merging: setup ## Run only the cluster merging job
	@echo "==> Running merging job"
	@DAGSTER_HOME=$(DAGSTER_HOME_DIR) $(PYTHON) -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j merging_job
	@echo "✓ Merging job completed"

.PHONY: run-job
run-job: setup ## Run a specific job with environment (usage: make run-job JOB=job_name ENV=prod)
	@echo "==> Running job $(JOB) in $(ENV) environment"
	@if [ -z "$(JOB)" ]; then \
		echo "Error: JOB parameter is required. Usage: make run-job JOB=job_name ENV=env_name"; \
		exit 1; \
	fi
	@DAGSTER_HOME=$(DAGSTER_HOME_DIR) $(PYTHON) -m $(PACKAGE_NAME) run $(JOB) --env $(ENV)
	@echo "✓ Job $(JOB) completed"

.PHONY: full-pipeline
full-pipeline: setup ## Run the full pipeline with specified environment (usage: make full-pipeline ENV=prod)
	@echo "==> Running full pipeline in $(ENV) environment"
	@DAGSTER_HOME=$(DAGSTER_HOME_DIR) $(PYTHON) -m $(PACKAGE_NAME) run full_pipeline_job --env $(ENV)
	@echo "✓ Full pipeline completed"

################################################################################
# MEMORY MANAGEMENT TARGETS
################################################################################

##@ Memory Management

.PHONY: run-memory-optimized
run-memory-optimized: setup ## Run a job with memory optimization settings (usage: make run-memory-optimized JOB=job_name)
	@echo "==> Running job with memory optimization"
	@if [ -z "$(JOB)" ]; then \
		echo "Error: JOB parameter is required. Usage: make run-memory-optimized JOB=job_name"; \
		exit 1; \
	fi
	@DAGSTER_HOME=$(DAGSTER_HOME_DIR) \
	 DAGSTER_MULTIPROCESS_MEMORY_OPTIMIZED=1 \
	 $(PYTHON) -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j $(JOB)
	@echo "✓ Memory-optimized job $(JOB) completed"

.PHONY: run-visualization
run-visualization: setup ## Run visualization job with memory optimization
	@echo "==> Running visualization job with memory optimization"
	@DAGSTER_HOME=$(DAGSTER_HOME_DIR) \
	 DAGSTER_MULTIPROCESS_MEMORY_OPTIMIZED=1 \
	 DAGSTER_MULTIPROCESS_CHUNK_SIZE=1 \
	 $(PYTHON) -m dagster job execute -m $(PACKAGE_NAME).dagster.definitions -j internal_visualization
	@echo "✓ Visualization job completed"

################################################################################
# DOCUMENTATION TARGETS
################################################################################

##@ Documentation

.PHONY: docs-deps
docs-deps: ## Install documentation dependencies
	@echo "==> Installing documentation dependencies"
	@uv add --group docs sphinx sphinx-rtd-theme sphinx-autodoc-typehints sphinx-autobuild
	@echo "✓ Documentation dependencies installed"

.PHONY: docs
docs: docs-deps ## Build documentation
	@echo "==> Building documentation"
	@cd $(DOCS_DIR) && LC_ALL=C.UTF-8 LANG=C.UTF-8 $(MAKE) html
	@echo "✓ Documentation built successfully. Open docs/build/html/index.html to view"

.PHONY: docs-server
docs-server: docs-deps ## Start documentation server (http://localhost:8000)
	@echo "==> Starting documentation server"
	@cd $(DOCS_DIR) && LC_ALL=C.UTF-8 LANG=C.UTF-8 $(PYTHON) -m sphinx_autobuild source build/html --port 8000 --host 0.0.0.0
	@echo "✓ Documentation server running at http://localhost:8000" 