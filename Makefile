# FinEasy AI Backend Makefile

.PHONY: help install dev test lint format clean docker-build docker-run docker-stop

# Default target
help:
	@echo "FinEasy AI Backend - Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  dev         - Run development server"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  clean       - Clean cache and temp files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run  - Run with Docker Compose"
	@echo "  docker-stop - Stop Docker containers"

# Install dependencies
install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

# Run development server
dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	pytest

# Run linting
lint:
	flake8 app/
	mypy app/

# Format code
format:
	black app/
	black tests/

# Clean cache and temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +

# Docker commands
docker-build:
	docker-compose build

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f ai-backend

# Development setup
setup-dev:
	python -m venv venv
	./venv/bin/pip install -r requirements.txt
	./venv/bin/python -m spacy download en_core_web_sm
	@echo "Virtual environment created. Activate with: source venv/bin/activate"