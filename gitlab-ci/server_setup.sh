#!/bin/bash
# Server setup script for Dagster pipeline deployment
# This script prepares a server for deployment of the Dagster pipeline

set -e  # Exit on any error

# Default values
DEPLOY_PATH="/opt/dagster/deploy"
DATA_PATH="/opt/dagster/data"
DAGSTER_HOME="/opt/dagster/dagster_home"
REGISTRY_URL=""
REGISTRY_USER=""
REGISTRY_PASSWORD=""
ENV="prod"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --deploy-path)
      DEPLOY_PATH="$2"
      shift 2
      ;;
    --data-path)
      DATA_PATH="$2"
      shift 2
      ;;
    --dagster-home)
      DAGSTER_HOME="$2"
      shift 2
      ;;
    --registry-url)
      REGISTRY_URL="$2"
      shift 2
      ;;
    --registry-user)
      REGISTRY_USER="$2"
      shift 2
      ;;
    --registry-password)
      REGISTRY_PASSWORD="$2"
      shift 2
      ;;
    --env)
      ENV="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Print configuration
echo "==> Server Setup Configuration:"
echo "Deploy Path: $DEPLOY_PATH"
echo "Data Path: $DATA_PATH"
echo "Dagster Home: $DAGSTER_HOME"
echo "Environment: $ENV"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "==> Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "==> Docker installed successfully"
else
    echo "==> Docker is already installed"
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "==> Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.18.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "==> Docker Compose installed successfully"
else
    echo "==> Docker Compose is already installed"
fi

# Create necessary directories
echo "==> Creating directories..."
sudo mkdir -p $DEPLOY_PATH
sudo mkdir -p $DATA_PATH/{internal,external,merging,raw}
sudo mkdir -p $DAGSTER_HOME

# Set appropriate permissions
echo "==> Setting permissions..."
sudo chown -R $USER:$USER $DEPLOY_PATH
sudo chown -R $USER:$USER $DATA_PATH
sudo chown -R $USER:$USER $DAGSTER_HOME

# Login to Docker registry if provided
if [ ! -z "$REGISTRY_URL" ] && [ ! -z "$REGISTRY_USER" ] && [ ! -z "$REGISTRY_PASSWORD" ]; then
    echo "==> Logging in to Docker registry..."
    echo "$REGISTRY_PASSWORD" | docker login $REGISTRY_URL -u $REGISTRY_USER --password-stdin
fi

# Copy configuration files to the deployment directory
echo "==> Creating environment file..."
cat > $DEPLOY_PATH/.env << EOF
# Dagster environment configuration
DAGSTER_HOME=$DAGSTER_HOME
DATA_DIR=$DATA_PATH
INTERNAL_DATA_DIR=$DATA_PATH/internal
EXTERNAL_DATA_DIR=$DATA_PATH/external
MERGING_DATA_DIR=$DATA_PATH/merging
ENV=$ENV
EOF

# Create docker-compose.yml in the deployment directory
echo "==> Creating docker-compose.yml..."
cat > $DEPLOY_PATH/docker-compose.yml << EOF
version: "3.8"

services:
  clustering:
    image: ${REGISTRY_URL:-registry.gitlab.com/your-org/your-project}/dagster-pipeline:latest
    env_file:
      - .env
    volumes:
      - $DATA_PATH:/app/data
      - $DEPLOY_PATH/logs:/app/logs
      - $DEPLOY_PATH/configs:/app/configs
      - $DAGSTER_HOME:/app/dagster_home
    environment:
      - DAGSTER_HOME=/app/dagster_home
      - PYTHONPATH=/app
    depends_on:
      postgres:
        condition: service_healthy

  # PostgreSQL database for Dagster
  postgres:
    image: postgres:13
    container_name: dagster-postgres
    environment:
      POSTGRES_USER: dagster
      POSTGRES_PASSWORD: dagster
      POSTGRES_DB: dagster
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: [ "CMD", "pg_isready", "-U", "dagster" ]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
EOF

echo "==> Server setup completed successfully!"
echo "The server is now ready for deployment."
echo ""
echo "To start the service:"
echo "  cd $DEPLOY_PATH"
echo "  docker-compose up -d"
echo ""
echo "To check the logs:"
echo "  docker-compose logs -f clustering" 