################################################################################
# STORE CLUSTERING - DASHBOARD MAKEFILE
################################################################################
#
# Author: Jackson Yang <Jackson.Yang@cvshealth.com>
# Version: 1.0.0
#
# Description:
#   This Makefile provides targets for developing and running the
#   Store Clustering dashboard.
#
################################################################################

##@ Project Configuration
################################################################################
# PROJECT CONFIGURATION
################################################################################

# Package information
PACKAGE_NAME := store-clustering-dashboard
VERSION := 1.0.0
AUTHOR := Jackson Yang

# Python command - use uv run for environment-specific configurations
PYTHON := uv run

# Directory paths
SRC_DIR := dashboard
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

.PHONY: help print-env version

help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\n\033[1mStore Clustering Dashboard Help\033[0m\n"} \
		/^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	@echo ""
	@echo "Usage examples:"
	@echo "  make install             # Install dependencies"
	@echo "  make dashboard           # Start the dashboard"
	@echo ""
	@echo "Environment variables priority:"
	@echo "  1. Command line (make dashboard DATA_DIR=/custom/path)"
	@echo "  2. Shell environment variables"
	@echo "  3. Variables from .env file"
	@echo "  4. Default values in Makefile"
	@echo ""
	@echo "Project info:"
	@echo "  Version: $(VERSION)"
	@echo "  Author:  $(AUTHOR)"
	@echo ""

print-env: ## Display current environment variable values
	@echo "==> Current environment configuration:"
	@echo "- DATA_DIR: $(DATA_DIR)"
	@echo "- ENV: $(ENV)"

version: ## Display version and author information
	@echo "=================================================="
	@echo "  $(PACKAGE_NAME) version $(VERSION)"
	@echo "  Developed by $(AUTHOR)"
	@echo "=================================================="

##@ Environment Setup
################################################################################
# ENVIRONMENT SETUP
################################################################################

.PHONY: install update

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
	@uv sync
	@echo "✓ All dependencies installed successfully"

update: ## Update all project dependencies to latest versions
	@echo "==> Updating all dependencies"
	@uv lock --upgrade && uv sync
	@echo "✓ Dependencies updated successfully"

##@ Development Tools
################################################################################
# DEVELOPMENT TOOLS
################################################################################

.PHONY: run

run: ## Run the clustering dashboard
	@echo "==> Starting Assortment Chatbot"
	@$(PYTHON) -m streamlit run src/chatbot/app.py --server.port 8501
	@echo "✓ Dashboard server stopped"

kill:
	@echo "==> Killing port 8501"
	@kill -9 $(shell lsof -t -i:8501)
	@echo "✓ Port 8501 killed"

##@ Code Quality & Testing
################################################################################
# CODE QUALITY & TESTING
################################################################################

.PHONY: format lint type-check check-all test

# Code Quality
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
	@$(PYTHON) -m mypy $(SRC_DIR)
	@echo "==> Running pyright type checker"
	-@$(PYTHON) -m pyright $(SRC_DIR)
	@echo "✓ Type checking complete (warnings may be present)"

check-all: format lint type-check ## Run all code quality checks 
	@echo "✓ All code quality checks completed successfully"

# Testing
test: ## Run tests for dashboard components
	@echo "==> Running tests for dashboard components"
	@$(PYTHON) -m pytest $(TESTS_DIR) -v
	@echo "✓ Tests completed"

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
