# Docker Deployment Guide

This guide explains how to use the Docker configuration for deploying and developing the Data Chat Assistant application.

## Overview

The Docker configuration consists of:

1. **Dockerfile**: Defines the application container image
2. **docker-compose.yml**: Orchestrates the application and required services

## Quick Start

To start the application using Docker:

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f dashboard

# Access the application
# Open http://localhost:8501 in your browser
```

## Services

The Docker configuration includes the following services:

### Dashboard (Main Application)

The main Data Chat Assistant application container, running the Streamlit dashboard.

- **Port**: 8501
- **Volumes**:
  - `app_data`: Persistent data storage
  - `app_logs`: Application logs
  - `./config`: Configuration files (read-only)

### MongoDB

Database for storing user sessions, query history, and feedback.

- **Port**: 27017
- **Volume**: `mongo_data` for persistent storage
- **Database**: `datachat`

## Environment Variables

The application can be configured using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `ENV` | Environment (dev, test, prod) | `dev` |
| `DATA_DIR` | Path to data directory | `/app/data` |
| `LOGS_DIR` | Path to logs directory | `/app/logs` |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `MONGODB_URI` | MongoDB connection string | `mongodb://mongo:27017/datachat` |

## Development Workflow

For development with Docker:

### Running with Local Changes

To see changes immediately without rebuilding:

```bash
# Start MongoDB service only
docker-compose up -d mongo

# Run the application locally with the MongoDB from Docker
# Set environment variables to match docker-compose.yml
export MONGODB_URI=mongodb://localhost:27017/datachat
make dashboard
```

### Running Tests

```bash
# Run tests inside Docker
docker-compose run --rm dashboard make test

# Run specific tests
docker-compose run --rm dashboard python -m pytest tests/test_specific.py
```

## Container Maintenance

Common maintenance commands:

```bash
# Rebuild containers
docker-compose build

# Stop and remove containers
docker-compose down

# Stop, remove containers, and delete volumes
docker-compose down -v

# View logs
docker-compose logs -f [service_name]

# Start a shell in a container
docker-compose exec dashboard bash
```

## Troubleshooting

### Container Won't Start

Check the logs:
```bash
docker-compose logs dashboard
```

Common issues:
1. **MongoDB connection error**: Ensure MongoDB container is running
2. **Permission issues**: Volumes might have incorrect permissions

### Reset Environment

To completely reset the environment:
```bash
docker-compose down -v
docker-compose up -d
```

## Production Deployment

For production deployment:

1. Set environment variable `ENV=prod` in docker-compose.yml
2. Consider using Docker Swarm or Kubernetes for orchestration
3. Set up proper authentication for MongoDB
4. Configure HTTPS with a reverse proxy like Nginx

Example production docker-compose override:

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  dashboard:
    environment:
      - ENV=prod
      - LOG_LEVEL=WARNING
    restart: always
    
  mongo:
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=securepassword
    restart: always
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/certs:/etc/nginx/certs
    depends_on:
      - dashboard
```

To use in production:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
``` 