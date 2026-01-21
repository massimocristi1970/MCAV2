.PHONY: help build run dev stop clean test lint format install

help: ## Show this help message
	@echo "Business Finance Scorecard - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

build: ## Build Docker image
	docker-compose build

run: ## Run application in production mode
	docker-compose up -d

dev: ## Run application in development mode
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

stop: ## Stop all services
	docker-compose down

clean: ## Clean up containers and images
	docker-compose down --rmi all --volumes --remove-orphans

test: ## Run tests
	python -m pytest tests/ -v --cov=app --cov-report=html

lint: ## Run linting
	flake8 app/
	mypy app/

format: ## Format code
	black app/
	isort app/

logs: ## View application logs
	docker-compose logs -f business-finance-app

shell: ## Access application shell
	docker-compose exec business-finance-app /bin/bash

restart: ## Restart application
	docker-compose restart business-finance-app

backup: ## Backup data
	tar -czf backup_$$(date +%Y%m%d_%H%M%S).tar.gz data/ logs/

health: ## Check application health
	curl -f http://localhost:8501/_stcore/health || echo "Application not healthy"
