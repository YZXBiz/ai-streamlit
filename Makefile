################################################################################
# PANDASAI CHATBOT APPLICATION MAKEFILE
################################################################################
#
# Description:
#   This Makefile provides targets for developing and running the
#   PandasAI chatbot application.
#
################################################################################

##@ Project Configuration
################################################################################
# PROJECT CONFIGURATION
################################################################################

# Python command - use uv run for environment-specific configurations
PYTHON := uv run

# Directory paths
SRC_DIR := src
TESTS_DIR := tests
LOGS_DIR := $(shell pwd)/logs

# Include .env file if it exists
# This loads environment variables from .env before setting defaults
-include .env

# Environment variables (defaults if not set in environment or .env)
DATA_DIR ?= $(shell pwd)/data
ENV ?= dev

# Set UV_LINK_MODE if not already defined, default to copy
UV_LINK_MODE ?= copy

# Export environment variables for child processes
export DATA_DIR
export UV_LINK_MODE

# Default target
.DEFAULT_GOAL := help

##@ Help & Information
################################################################################
# HELP & INFORMATION
################################################################################

.PHONY: help print-env

help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\n\033[1mPandasAI Chatbot Help\033[0m\n"} \
		/^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	@echo ""
	@echo "Usage examples:"
	@echo "  make install             # Install dependencies"
	@echo "  make run                 # Start the PandasAI application"
	@echo ""
	@echo "Environment variables priority:"
	@echo "  1. Command line (make run DATA_DIR=/custom/path)"
	@echo "  2. Shell environment variables"
	@echo "  3. Variables from .env file"
	@echo "  4. Default values in Makefile"
	@echo ""

print-env: ## Display current environment variable values
	@echo "==> Current environment configuration:"
	@echo "- DATA_DIR: $(DATA_DIR)"
	@echo "- ENV: $(ENV)"

##@ Environment Setup
################################################################################
# ENVIRONMENT SETUP
################################################################################

.PHONY: install update

install: ## Install project dependencies using uv
	@echo "==> Checking if uv is installed..."
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Installing uv package manager..."; \
		curl -Ls https://astral.sh/uv/install.sh | bash; \
		echo "✓ uv installed successfully"; \
	else \
		echo "✓ uv is already installed"; \
	fi
	@echo "==> Installing project dependencies"
	@uv sync
	@echo "✓ All dependencies installed successfully"

update: ## Update all project dependencies to latest versions
	@echo "==> Updating all dependencies"
	@uv lock --upgrade && uv sync
	@echo "✓ Dependencies updated successfully"

##@ Application
################################################################################
# APPLICATION COMMANDS
################################################################################

.PHONY: run run-dev run-backend kill

run: ## Run the PandasAI frontend application
	@echo "==> Starting PandasAI Frontend"
	@$(PYTHON) -m streamlit run frontend/app.py --server.port 8503
	@echo "✓ PandasAI Frontend stopped"

run-dev: ## Run the application in development mode with auto-reload
	@echo "==> Starting application in development mode"
	@$(PYTHON) -m streamlit run frontend/app.py --server.port 8503 --server.headless false --server.runOnSave true
	@echo "✓ Application stopped"

run-backend: ## Run the PandasAI Backend API
	@echo "==> Starting PandasAI Backend API"
	@cd backend && $(PYTHON) -m app.main
	@echo "✓ Backend API stopped"

kill: ## Kill processes on specific ports
	@echo "==> Killing port 8503"
	@kill -9 $$(lsof -t -i:8503) 2>/dev/null || echo "No process running on port 8503"
	@echo "✓ Port 8503 killed"

##@ Code Quality & Testing
################################################################################
# CODE QUALITY & TESTING
################################################################################

.PHONY: format lint type-check check-all test test-backend test-backend-integration test-frontend test-all clean

# Code Quality
format: ## Format code with ruff formatter
	@echo "==> Formatting code with ruff"
	@$(PYTHON) -m ruff format $(SRC_DIR) $(TESTS_DIR) frontend backend
	@echo "✓ Code formatting complete"

lint: ## Lint code and auto-fix issues where possible
	@echo "==> Linting code with ruff"
	@$(PYTHON) -m ruff check $(SRC_DIR) $(TESTS_DIR) frontend backend --fix
	@echo "✓ Code linting complete"

type-check: ## Run type checking with mypy and pyright
	@echo "==> Running mypy type checker"
	@$(PYTHON) -m mypy $(SRC_DIR) frontend backend
	@echo "==> Running pyright type checker"
	-@$(PYTHON) -m pyright $(SRC_DIR) frontend backend
	@echo "✓ Type checking complete (warnings may be present)"

check-all: format lint type-check ## Run all code quality checks
	@echo "✓ All code quality checks completed successfully"

# Testing
test: ## Run tests for common components
	@echo "==> Running tests for common components"
	@$(PYTHON) -m pytest $(TESTS_DIR) -v
	@echo "✓ Tests completed"

test-backend: ## Run tests for the backend
	@echo "==> Running tests for backend components"
	@$(PYTHON) -m pytest backend/tests -v
	@echo "✓ Tests completed"

test-backend-integration: ## Run integration tests for the backend
	@echo "==> Running integration tests for backend components"
	@mkdir -p backend/reports/coverage
	@cd backend && python -m pytest tests/test_health.py tests/test_auth.py tests/test_files.py tests/test_chat.py -v --cov=tests --cov-report=term --cov-report=html:reports/coverage
	@echo "✓ Integration tests completed"
	@echo "Coverage report available at: backend/reports/coverage/index.html"

test-backend-coverage: ## Run backend tests with coverage reporting
	@echo "==> Running backend tests with coverage"
	@mkdir -p reports/coverage
	@$(PYTHON) -m pytest backend/tests -v --cov=backend --cov-report=term --cov-report=html:reports/coverage
	@echo "✓ Tests completed with coverage report"

test-frontend: ## Run tests for the frontend
	@echo "==> Running tests for frontend components"
	@$(PYTHON) -m pytest frontend/tests -v
	@echo "✓ Tests completed"

test-all: test test-backend test-frontend ## Run all tests
	@echo "✓ All tests completed"

validate-tests: ## Run all tests and verify they're working properly
	@echo "==> Running comprehensive test validation"
	@echo "Running individual test files..."
	@cd backend && python tests/run_tests.py
	@echo "\nRunning test suite with coverage..."
	@make test-backend-integration
	@echo "\n✓ All tests validated successfully"

clean: ## Clean Python cache files
	@echo "==> Cleaning cache files"
	@rm -rf .pytest_cache
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.pyd" -delete
	@echo "✓ Cache files cleaned"

##@ Documentation
################################################################################
# DOCUMENTATION
################################################################################

.PHONY: docs docs-clean

docs: ## Build Sphinx documentation
	@echo "==> Building Sphinx documentation"
	@mkdir -p docs/_build/html
	@$(PYTHON) -m sphinx.cmd.build -b html docs docs/_build/html
	@echo "✓ Documentation built successfully"
	@echo "   Open docs/_build/html/index.html in your browser to view"

docs-clean: ## Clean documentation build files
	@echo "==> Cleaning documentation build files"
	@rm -rf docs/_build
	@echo "✓ Documentation cleaned"
