.PHONY: install run clean lint-fix

# Default Python interpreter
PYTHON := uv run
PIP := uv pip

# Install dependencies
install:
	$(PIP) install -e .
	$(PIP) install pygwalker pandas streamlit

# Run the PyGWalker application
run:
	$(PYTHON) -m clustering.dashboard.app

# Run the application in debug mode
run-debug:
	PYTHONPATH=. $(PYTHON) -m clustering.dashboard.app --debug

# Clean temporary files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/

# Create requirements file from current environment
requirements:
	$(PIP) freeze > requirements.txt

# Install pre-commit hooks
setup-dev:
	$(PIP) install pre-commit pytest mypy ruff
	pre-commit install

# Run tests
test:
	pytest

# Run linting
lint:
	ruff check .
	mypy .

# Fix linting issues
lint-fix:
	ruff check --fix . 