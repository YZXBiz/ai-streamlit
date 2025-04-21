# Docker Deployment Guide

This guide explains how to manually deploy the Dagster pipeline using Docker and docker-compose.

## Prerequisites

- Docker (20.10+)
- docker-compose (2.0+)
- Access to the Dagster pipeline source code or Docker image

## Server Setup

You can set up the server manually or use the provided setup script:

```bash
# Using the setup script
chmod +x gitlab-ci/server_setup.sh
./gitlab-ci/server_setup.sh --env prod
```

## Manual Deployment

### 1. Create Required Directories

```bash
# Create deployment directories
mkdir -p /opt/dagster/deploy
mkdir -p /opt/dagster/data/{internal,external,merging}
mkdir -p /opt/dagster/dagster_home
```

### 2. Create Environment File

Create a `.env` file in the deployment directory:

```bash
# Create environment file
cat > /opt/dagster/deploy/.env << EOF
# Dagster environment configuration
DAGSTER_HOME=/opt/dagster/dagster_home
DATA_DIR=/opt/dagster/data
INTERNAL_DATA_DIR=/opt/dagster/data/internal
EXTERNAL_DATA_DIR=/opt/dagster/data/external
MERGING_DATA_DIR=/opt/dagster/data/merging
ENV=prod
EOF
```

### 3. Create docker-compose.yml

Create a `docker-compose.yml` file in the deployment directory:

```bash
cat > /opt/dagster/deploy/docker-compose.yml << EOF
version: "3.8"

services:
  clustering:
    image: registry.gitlab.com/your-org/your-project/dagster-pipeline:latest
    env_file:
      - .env
    volumes:
      - /opt/dagster/data:/app/data
      - /opt/dagster/logs:/app/logs
      - /opt/dagster/configs:/app/configs
      - /opt/dagster/dagster_home:/app/dagster_home
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
```

### 4. Start Services

```bash
# Login to GitLab Docker registry (if needed)
docker login registry.gitlab.com

# Go to deployment directory
cd /opt/dagster/deploy

# Start services in detached mode
docker-compose up -d
```

### 5. Verify Deployment

```bash
# Check that services are running
docker-compose ps

# Check logs for any issues
docker-compose logs -f clustering
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DAGSTER_HOME` | Dagster home directory | /app/dagster_home |
| `DATA_DIR` | Base data directory | /app/data |
| `INTERNAL_DATA_DIR` | Internal data directory | /app/data/internal |
| `EXTERNAL_DATA_DIR` | External data directory | /app/data/external |
| `MERGING_DATA_DIR` | Merging data directory | /app/data/merging |
| `ENV` | Environment name (dev, staging, prod) | prod |

### Custom Configuration

To use a custom configuration:

1. Copy your configuration files to the server:
   ```bash
   scp configs/prod.yml user@server:/opt/dagster/configs/
   ```

2. Update the docker-compose.yml to use this configuration:
   ```yaml
   services:
     clustering:
       # ...
       command: configs/prod.yml
   ```

## Maintenance

### Updating the Pipeline

```bash
# Pull the latest image
docker-compose pull

# Restart the services
docker-compose down
docker-compose up -d
```

### Backup and Restore

```bash
# Backup data
tar -czvf dagster_backup.tar.gz /opt/dagster/data /opt/dagster/dagster_home

# Restore data
tar -xzvf dagster_backup.tar.gz -C /
```

## Troubleshooting

### Common Issues

1. **Container fails to start**:
   - Check logs: `docker-compose logs clustering`
   - Verify volumes are mounted correctly
   - Ensure permissions are set correctly

2. **Database connection errors**:
   - Check if PostgreSQL is running: `docker-compose ps postgres`
   - Verify PostgreSQL credentials in docker-compose.yml
   - Check if the database was initialized correctly

3. **File permission errors**:
   - Ensure proper ownership: `sudo chown -R $USER:$USER /opt/dagster`
   - Check volume mounts in docker-compose.yml 