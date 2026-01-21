#!/bin/bash

# Business Finance Application Startup Script

set -e

echo "🚀 Starting Business Finance Scorecard..."

# Check if environment file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found. Creating from template..."
    cp env_example.env .env
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
