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
