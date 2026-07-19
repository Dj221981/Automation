.PHONY: install test lint format security build clean all

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src

lint:
	black --check src/ tests/
	flake8 src/ tests/
	isort --check-only src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

security:
	bandit -r src/
	safety check

build:
	npm run build

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name "dist" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

all: lint security test build
