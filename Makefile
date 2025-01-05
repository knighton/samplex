.PHONY: help install clean lint test coverage fix format check docs security benchmark profile deps all

PY = python3
VENV = .venv
BIN = $(VENV)/bin
SRC = samplex
TESTS = tests

help:
	@echo "Available commands:"
	@echo "  make install    - Install development dependencies"
	@echo "  make clean      - Remove build/test artifacts"
	@echo "  make lint       - Run all linters"
	@echo "  make test       - Run tests"
	@echo "  make coverage   - Run tests with coverage"
	@echo "  make fix        - Auto-fix code style issues"
	@echo "  make format     - Run code formatters"
	@echo "  make check      - Run all checks (lint + test)"
	@echo "  make docs       - Generate documentation"
	@echo "  make security   - Run security checks"
	@echo "  make benchmark  - Run performance benchmarks"
	@echo "  make profile    - Run profiling"
	@echo "  make deps       - Generate dependency graph"
	@echo "  make all        - Run clean install check"

clean-venv:
	rm -rf $(VENV)

install: clean-venv
	$(PY) -m venv $(VENV)
	$(BIN)/python3 -m pip install --upgrade pip setuptools wheel
	$(BIN)/pip install -e ".[dev]"
	$(BIN)/pre-commit install

clean: clean-venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -wholename "*/docs/_build" -exec rm -rf {} +

lint:
	$(BIN)/pre-commit run --all-files

test:
	$(BIN)/pytest $(TESTS) -v

coverage:
	$(BIN)/pytest $(TESTS) -v --cov=$(SRC) --cov-report=html --cov-report=term

fix:
	$(BIN)/ruff check --fix $(SRC) $(TESTS)

format:
	$(BIN)/ruff format $(SRC) $(TESTS)

check: lint test

docs:
	$(BIN)/sphinx-build -b html docs docs/_build/html

security:
	$(BIN)/bandit -r $(SRC)
	$(BIN)/safety check

#benchmark:
#	$(BIN)/pytest $(TESTS)/benchmarks --benchmark-only
#
#profile:
#	$(BIN)/python -m cProfile -o profile.stats $(SRC)/main.py
#	$(BIN)/snakeviz profile.stats

deps:
	$(BIN)/pydeps $(SRC) --max-bacon=2 --cluster --no-show

all: clean install check
