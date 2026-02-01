# Makefile for gpssi project
# Modern Python project management (2026)

.PHONY: all help venv install install-dev install-all lint format typecheck test test-cov test-benchmark examples clean clean-all

# Python interpreter to use
PYTHON ?= python3
# Virtual environment directory
VENV_DIR ?= .venv
# Activate command based on OS
ifeq ($(OS),Windows_NT)
    VENV_ACTIVATE := $(VENV_DIR)/Scripts/activate
    ACTIVATE := . $(VENV_ACTIVATE) &&
else
    VENV_ACTIVATE := $(VENV_DIR)/bin/activate
    ACTIVATE := . $(VENV_ACTIVATE) &&
endif

# Default target
all: venv install-dev lint typecheck test

# Help target
help:
	@echo "gpssi - Gaussian Process Sampling of Segmentation Images"
	@echo ""
	@echo "Available targets:"
	@echo "  venv           Create virtual environment"
	@echo "  install        Install package in production mode"
	@echo "  install-dev    Install package in development mode with dev dependencies"
	@echo "  install-all    Install package with all optional dependencies"
	@echo "  lint           Run linter (ruff)"
	@echo "  format         Format code (ruff format)"
	@echo "  typecheck      Run type checker (mypy)"
	@echo "  test           Run tests"
	@echo "  test-cov       Run tests with coverage report"
	@echo "  test-benchmark Run performance benchmarks"
	@echo "  examples       Run example scripts"
	@echo "  clean          Remove temporary files and caches"
	@echo "  clean-all      Remove everything including venv"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON         Python interpreter (default: python3)"
	@echo "  VENV_DIR       Virtual environment directory (default: .venv)"

# Create virtual environment
venv:
	@echo "Creating virtual environment in $(VENV_DIR)..."
	$(PYTHON) -m venv $(VENV_DIR)
	$(ACTIVATE) pip install --upgrade pip setuptools wheel
	@echo "Virtual environment created. Activate with: source $(VENV_ACTIVATE)"

# Install package in production mode
install: venv
	@echo "Installing package..."
	$(ACTIVATE) pip install -e .

# Install package in development mode
install-dev: venv
	@echo "Installing package with development dependencies..."
	$(ACTIVATE) pip install -e ".[dev]"
	$(ACTIVATE) pre-commit install

# Install package with all dependencies
install-all: venv
	@echo "Installing package with all dependencies..."
	$(ACTIVATE) pip install -e ".[all]"
	$(ACTIVATE) pre-commit install

# Run linter
lint:
	@echo "Running linter..."
	$(ACTIVATE) ruff check src/ tests/ examples/

# Format code
format:
	@echo "Formatting code..."
	$(ACTIVATE) ruff format src/ tests/ examples/
	$(ACTIVATE) ruff check --fix src/ tests/ examples/

# Run type checker
typecheck:
	@echo "Running type checker..."
	$(ACTIVATE) mypy src/gpssi

# Run tests
test:
	@echo "Running tests..."
	$(ACTIVATE) pytest tests/ -v

# Run tests with coverage
test-cov:
	@echo "Running tests with coverage..."
	$(ACTIVATE) pytest tests/ -v --cov=src/gpssi --cov-report=html --cov-report=term-missing

# Run performance benchmarks
test-benchmark:
	@echo "Running performance benchmarks..."
	$(ACTIVATE) pytest tests/ -v --benchmark-only --benchmark-autosave --benchmark-compare

# Run examples
examples: install
	@echo "Running 2D example..."
	$(ACTIVATE) cd examples && $(PYTHON) gpssi_example.py
	@echo "Note: 3D example requires SimpleITK which is not included by default"

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.py[cod]" -delete
	find . -type f -name "*~" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".benchmarks" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."

# Clean everything including virtual environment
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Clean-all complete."
