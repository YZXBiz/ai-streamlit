#!/bin/bash
set -e

# Configuration
ACR_NAME="your-acr-name" # Replace with your Azure Container Registry name
IMAGE_NAME="flat-chatbot"
IMAGE_TAG="latest"

# Login to Azure
echo "Logging in to Azure..."
az login

# Login to ACR
echo "Logging in to Azure Container Registry..."
az acr login --name $ACR_NAME

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME:$IMAGE_TAG .

# Tag the image for ACR
echo "Tagging image for ACR..."
docker tag $IMAGE_NAME:$IMAGE_TAG $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG

# Push the image to ACR
echo "Pushing image to ACR..."
docker push $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG

echo "Successfully deployed $IMAGE_NAME:$IMAGE_TAG to $ACR_NAME.azurecr.io"
echo "You can now create an Azure Container Instance or Azure App Service to run this container."
