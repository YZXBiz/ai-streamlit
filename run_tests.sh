#!/bin/bash

# Run tests with coverage
uv run -m pytest tests/ --cov=app --cov-report=term-missing --cov-report=html:coverage_html
