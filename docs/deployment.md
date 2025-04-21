# Deployment Guide for Dagster Pipeline

This guide explains how to deploy the Dagster pipeline using GitLab CI/CD.

## Prerequisites

Before setting up the deployment pipeline, ensure you have:

1. A GitLab account with access to create projects and repositories
2. Servers or cloud resources for each environment (dev, staging, prod)
3. SSH access to deployment servers
4. Docker and docker-compose installed on all deployment servers

## GitLab CI/CD Setup

### CI/CD Variables

Set up the following CI/CD variables in GitLab (Settings → CI/CD → Variables):

#### Authentication Variables

| Variable Name | Description | Type |
|---------------|-------------|------|
| `CI_REGISTRY_USER` | GitLab registry username | Variable |
| `CI_REGISTRY_PASSWORD` | GitLab registry password | Variable |
| `DEV_SSH_PRIVATE_KEY` | SSH private key for dev server | File |
| `STAGING_SSH_PRIVATE_KEY` | SSH private key for staging server | File |
| `PROD_SSH_PRIVATE_KEY` | SSH private key for production server | File |

#### Deployment Server Variables

| Variable Name | Description | Default |
|---------------|-------------|---------|
| `DEV_SERVER_HOST` | Hostname for dev server | dev-dagster.example.com |
| `DEV_SERVER_USER` | SSH user for dev server | dagster |
| `DEV_DEPLOY_PATH` | Deployment path on dev server | /opt/dagster/deploy |
| `STAGING_SERVER_HOST` | Hostname for staging server | staging-dagster.example.com |
| `STAGING_SERVER_USER` | SSH user for staging server | dagster |
| `STAGING_DEPLOY_PATH` | Deployment path on staging server | /opt/dagster/deploy |
| `PROD_SERVER_HOST` | Hostname for production server | prod-dagster.example.com |
| `PROD_SERVER_USER` | SSH user for production server | dagster |
| `PROD_DEPLOY_PATH` | Deployment path on production server | /opt/dagster/deploy |

### Environment Setup on Servers

For each deployment server (dev, staging, prod), perform the following setup:

```bash
# Login to the server
ssh user@server

# Create deployment directory
sudo mkdir -p /opt/dagster/deploy
sudo mkdir -p /opt/dagster/data/{internal,external,merging}
sudo mkdir -p /opt/dagster/dagster_home

# Set permissions
sudo chown -R $USER:$USER /opt/dagster

# Install Docker and docker-compose
# [Instructions depend on server OS]

# Create a basic docker-compose.yml in the deployment directory
cat > /opt/dagster/deploy/docker-compose.yml << EOF
version: "3.8"

services:
  clustering:
    image: ${GITLAB_REGISTRY}/your-project/dagster-pipeline:latest
    env_file:
      - .env
    volumes:
      - /opt/dagster/data:/app/data
      - /opt/dagster/logs:/app/logs
      - /opt/dagster/configs:/app/configs
      - /opt/dagster/dagster_home:/app/dagster_home
    ports:
      - "3000:3000"
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
    ports:
      - "5432:5432"
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

## Deployment Process

### Manual Deployment

1. Go to the GitLab project → CI/CD → Pipelines
2. Find the latest successful pipeline on the main branch
3. Click on the "play" button next to the `deploy_dev`, `deploy_staging`, or `deploy_prod` job
4. Monitor the job logs for deployment status

### Automatic Deployment

- **Development**: Automatically triggered on pushes to branches (requires manual approval)
- **Staging**: Automatically triggered on merges to the main branch (requires manual approval)
- **Production**: Manually triggered for tagged releases matching the pattern `v*.*.*`

### Promoting Between Environments

1. Deploy to development environment and test
2. Merge to main branch and deploy to staging
3. Create a version tag (e.g., `v1.2.0`) to deploy to production

## Monitoring and Troubleshooting

### Checking Deployment Status

```bash
# SSH into the server
ssh user@server

# Go to the deployment directory
cd /opt/dagster/deploy

# Check running containers
docker-compose ps

# View container logs
docker-compose logs -f clustering
```

### Common Issues

1. **Image Pull Failure**: Check GitLab registry permissions and server internet connection
2. **Container Exit**: Check logs with `docker-compose logs clustering`
3. **Permission Issues**: Ensure proper volume permissions on the server

## Rollback Procedure

To rollback to a previous version:

```bash
# SSH into the server
ssh user@server

# Go to the deployment directory
cd /opt/dagster/deploy

# Pull a specific version
docker-compose down
export GITLAB_REGISTRY=registry.gitlab.com
docker pull ${GITLAB_REGISTRY}/your-project/dagster-pipeline:v1.1.0
docker tag ${GITLAB_REGISTRY}/your-project/dagster-pipeline:v1.1.0 ${GITLAB_REGISTRY}/your-project/dagster-pipeline:latest
docker-compose up -d
```

## Security Considerations

1. Use SSH keys instead of passwords for deployment
2. Store sensitive information in GitLab CI/CD variables
3. Do not hardcode credentials in docker-compose files or Dockerfiles
4. Regularly update the base Docker images to include security patches
5. Consider using GitLab Environments for deployment approval processes 