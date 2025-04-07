# Environment Variable Configuration

This document explains how environment variables are used in the clustering pipeline application and how to configure them correctly.

## Overview

The application uses environment variables to store sensitive configuration like API keys, passwords, and connection strings. These variables are referenced in the YAML configuration files using the `${VARIABLE_NAME}` syntax.

## How It Works

1. **Environment Variables Loading**:

   - Environment variables are loaded from `.env` files when the application starts
   - The application first looks for an environment-specific file (e.g., `.env.dev`, `.env.staging`, `.env.prod`)
   - If not found, it falls back to the default `.env` file

2. **Variable Substitution**:
   - Configuration files use the syntax `${VARIABLE_NAME}` to reference environment variables
   - Dagster automatically substitutes these references with the actual values at runtime
   - This keeps sensitive information out of the codebase and configuration files

## Setup Instructions

### 1. Create Environment Files

Create one or more of these files in the project root:

- `.env` - Default environment file
- `.env.dev` - Development environment-specific variables
- `.env.staging` - Staging environment-specific variables
- `.env.prod` - Production environment-specific variables

### 2. Add Required Variables

In your `.env` file, define the following variables as needed:

```
# Azure Storage credentials
AZURE_STORAGE_ACCOUNT=""
AZURE_STORAGE_KEY=""
AZURE_STORAGE_CONNECTION_STRING=""
AZURE_TENANT_ID=""
AZURE_CLIENT_ID=""
AZURE_CLIENT_SECRET=""

# Azure Blob Storage
ACCOUNT_URL=""
CONTAINER_NAME=""

# Snowflake credentials
SNOWFLAKE_ACCOUNT=""
SNOWFLAKE_USER=""
SNOWFLAKE_PASSWORD=""
SNOWFLAKE_WAREHOUSE=""
SNOWFLAKE_DATABASE=""

# Notification settings
SLACK_WEBHOOK_URL=""
```

### 3. Dependencies

The application uses `python-dotenv` to load environment variables from files. Install it with:

```bash
pip install python-dotenv
```

## Using Environment Variables in YAML Configuration

Environment variables are referenced in YAML configuration files using the `${VARIABLE_NAME}` syntax:

```yaml
readers:
  external_sales:
    kind: "SnowflakeReader"
    config:
      query: SELECT * FROM PROD_CLUSTERING_DB.RAW.EXTERNAL_SALES
      account: ${SNOWFLAKE_ACCOUNT}
      password: ${SNOWFLAKE_PASSWORD}
      user: ${SNOWFLAKE_USER}
```

## Security Considerations

- Never commit `.env` files to version control
- Add `.env*` to your `.gitignore` file
- Use different environments for different stages (dev/staging/prod)
- Keep production credentials separate from development
- Consider using a secrets manager for production deployments

## Troubleshooting

If environment variables are not being properly substituted:

1. Check that you have installed `python-dotenv`
2. Verify that your `.env` file is properly formatted
3. Ensure the variable names match between the `.env` file and the reference in the YAML
4. Run in verbose mode to see which environment files are being loaded
