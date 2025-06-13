# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

---

# docker-compose.yml
version: '3.8'

services:
  business-finance-app:
    build: .
    container_name: business-finance-scorecard
    ports:
      - "8501:8501"
    environment:
      # Application settings
      - APP_NAME=Business Finance Scorecard
      - APP_VERSION=2.0.0
      - DEBUG=false
      - LOG_LEVEL=INFO
      
      # Plaid API (set your actual values)
      - PLAID_CLIENT_ID=${PLAID_CLIENT_ID}
      - PLAID_SECRET=${PLAID_SECRET}
      - PLAID_ENV=production
      
      # Cache settings
      - CACHE_TTL=3600
      
      # Security
      - SECRET_KEY=${SECRET_KEY}
      
      # Database (SQLite for simplicity)
      - DATABASE_URL=sqlite:///./data/business_finance.db
      
      # Model paths
      - MODEL_PATH=app/models/model_artifacts/model.pkl
      - SCALER_PATH=app/models/model_artifacts/scaler.pkl
    
    volumes:
      # Persist data and logs
      - ./data:/app/data
      - ./logs:/app/logs
      # Mount model files
      - ./model.pkl:/app/app/models/model_artifacts/model.pkl
      - ./scaler.pkl:/app/app/models/model_artifacts/scaler.pkl
    
    restart: unless-stopped
    
    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '0.5'
          memory: 1G
    
    # Health check
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Redis for advanced caching
  redis:
    image: redis:7-alpine
    container_name: business-finance-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

  # Optional: PostgreSQL for production database
  postgres:
    image: postgres:15-alpine
    container_name: business-finance-db
    environment:
      - POSTGRES_DB=business_finance
      - POSTGRES_USER=app_user
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:

networks:
  default:
    name: business-finance-network

---

# .dockerignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# Streamlit
.streamlit/

# Logs
logs/
*.log

# Data files
data/
*.csv
*.json
*.xlsx
*.xls

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Git
.git/
.gitignore

# Documentation
docs/_build/
*.md

# Model files (should be explicitly mounted)
*.pkl
*.joblib
*.h5
*.pb

# Temporary files
tmp/
temp/
*.tmp

# Node modules (if any JS tools)
node_modules/

---

# docker-compose.dev.yml - Development configuration
version: '3.8'

services:
  business-finance-app:
    build: 
      context: .
      dockerfile: Dockerfile
    container_name: business-finance-scorecard-dev
    ports:
      - "8501:8501"
    environment:
      - DEBUG=true
      - LOG_LEVEL=DEBUG
      - STREAMLIT_SERVER_RELOAD_ON_CHANGE=true
    volumes:
      # Mount source code for development
      - .:/app
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    
    # Override command for development
    command: ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=true"]

---

# scripts/start.sh - Startup script
#!/bin/bash

# Business Finance Application Startup Script

set -e

echo "🚀 Starting Business Finance Scorecard..."

# Check if environment file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration"
fi

# Create necessary directories
mkdir -p data logs

# Check if model files exist
if [ ! -f "model.pkl" ]; then
    echo "❌ Error: model.pkl not found"
    echo "Please ensure ML model files are in the root directory"
    exit 1
fi

if [ ! -f "scaler.pkl" ]; then
    echo "❌ Error: scaler.pkl not found"
    echo "Please ensure ML model files are in the root directory"
    exit 1
fi

# Start the application
if [ "$1" = "dev" ]; then
    echo "🔧 Starting in development mode..."
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
else
    echo "🏭 Starting in production mode..."
    docker-compose up --build -d
    echo "✅ Application started successfully!"
    echo "🌐 Access at: http://localhost:8501"
    echo "📋 To view logs: docker-compose logs -f business-finance-app"
    echo "🛑 To stop: docker-compose down"
fi

---

# scripts/deploy.sh - Deployment script
#!/bin/bash

# Business Finance Application Deployment Script

set -e

echo "🚀 Deploying Business Finance Scorecard..."

# Configuration
IMAGE_NAME="business-finance-scorecard"
CONTAINER_NAME="business-finance-app"
PORT=8501

# Build the image
echo "🔨 Building Docker image..."
docker build -t $IMAGE_NAME .

# Stop existing container if running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "🛑 Stopping existing container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Run the new container
echo "▶️  Starting new container..."
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8501 \
    --env-file .env \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/model.pkl:/app/app/models/model_artifacts/model.pkl \
    -v $(pwd)/scaler.pkl:/app/app/models/model_artifacts/scaler.pkl \
    --restart unless-stopped \
    $IMAGE_NAME

echo "✅ Deployment completed successfully!"
echo "🌐 Application is available at: http://localhost:$PORT"
echo "📋 To view logs: docker logs -f $CONTAINER_NAME"
echo "🛑 To stop: docker stop $CONTAINER_NAME"

---

# Makefile - Development commands
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
	tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz data/ logs/

migrate: ## Run database migrations (if applicable)
	docker-compose exec business-finance-app python -c "from app.database import create_tables; create_tables()"

health: ## Check application health
	curl -f http://localhost:8501/_stcore/health || echo "Application not healthy"

---

# .github/workflows/ci.yml - GitHub Actions CI/CD
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov flake8 mypy black isort
    
    - name: Lint with flake8
      run: |
        flake8 app/ --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 app/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Type check with mypy
      run: |
        mypy app/ --ignore-missing-imports
    
    - name: Test with pytest
      run: |
        pytest tests/ -v --cov=app --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run security checks
      run: |
        pip install bandit safety
        bandit -r app/
        safety check --json

  build-and-deploy:
    needs: [test, security]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: |
        docker build -t business-finance-scorecard:latest .
    
    - name: Run Docker container for testing
      run: |
        docker run -d --name test-container -p 8501:8501 business-finance-scorecard:latest
        sleep 30
        curl -f http://localhost:8501/_stcore/health || exit 1
        docker stop test-container
    
    - name: Deploy to staging (if applicable)
      if: github.ref == 'refs/heads/develop'
      run: |
        echo "Deploy to staging environment"
        # Add your staging deployment commands here
    
    - name: Deploy to production
      if: github.ref == 'refs/heads/main'
      run: |
        echo "Deploy to production environment"
        # Add your production deployment commands here

---

# pytest.ini - Test configuration
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --strict-markers
    --strict-config
    --cov=app
    --cov-branch
    --cov-report=term-missing:skip-covered
    --cov-report=html:htmlcov
    --cov-report=xml
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
    api: API tests

---

# .pre-commit-config.yaml - Pre-commit hooks
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=127, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
        additional_dependencies: [types-all]

  - repo: https://github.com/pycqa/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-r', 'app/', '-x', 'tests/']

---

# setup.py - Package setup
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="business-finance-scorecard",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced Business Finance Analysis & Risk Assessment Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/business-finance-scorecard",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "prod": [
            "gunicorn>=21.0.0",
            "redis>=4.0.0",
            "postgresql>=0.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "business-finance-app=app.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "app": [
            "models/model_artifacts/*.pkl",
            "config/*.json",
            "static/*",
            "templates/*",
        ],
    },
)

---

# .gitignore - Updated Git ignore file
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# poetry
poetry.lock

# pdm
.pdm.toml

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.env.local
.env.development.local
.env.test.local
.env.production.local
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# Streamlit
.streamlit/

# Application specific
logs/
data/
*.pkl
*.joblib
*.h5
*.pb
backup_*.tar.gz

# IDE specific
.vscode/
.idea/
*.swp
*.swo
*~

# OS specific
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Docker
.dockerignore

# Temporary files
*.tmp
tmp/
temp/

# Documentation builds
docs/build/
docs/source/_static/
docs/source/_templates/

# Local configuration
config/local.py
config/production.py
secrets/

# Database files
*.db
*.sqlite
*.sqlite3

# Model artifacts (should be managed separately)
models/trained/
models/checkpoints/
experiments/

# Data files (should not be in version control)
data/raw/
data/processed/
data/external/
data/interim/

# Reports and outputs
reports/
outputs/
results/

# Jupyter notebook checkpoints
.ipynb_checkpoints

# Machine learning experiments
mlruns/
.mlflow/

# Configuration management
ansible/group_vars/production.yml
ansible/host_vars/

# Terraform
*.tfstate
*.tfstate.*
.terraform/

# Kubernetes
k8s/secrets/

# Certificates and keys
*.pem
*.key
*.crt
*.p12
*.pfx

# Application logs
access.log
error.log
app.log
debug.log