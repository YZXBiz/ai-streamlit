# Secure Secret Management

This guide explains how to securely manage secrets in the dashboard application using a combination of Azure Key Vault and Streamlit secrets.

## Architecture

Our secret management system follows a layered approach:

1. **Azure Key Vault**: The primary storage for sensitive secrets like API keys and passwords
2. **Streamlit Secrets**: Stores only Azure Key Vault connection details
3. **Environment Variables**: Act as the bridge between different configuration systems
4. **Pydantic Settings**: Provides type validation and default values

This architecture provides enterprise-grade security while maintaining a smooth developer experience.

## Setup Instructions

### 1. Create an Azure Key Vault

If you don't already have an Azure Key Vault, create one:

```bash
# Install the Azure CLI if you haven't already
# https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# Log in to Azure
az login

# Create a resource group (if you don't already have one)
az group create --name MyResourceGroup --location eastus

# Create a Key Vault
az keyvault create --name MyKeyVault --resource-group MyResourceGroup --location eastus
```

### 2. Store Your Secrets in Azure Key Vault

Add your sensitive secrets to the vault:

```bash
# Add OpenAI API key
az keyvault secret set --vault-name MyKeyVault --name OPENAI_API_KEY --value "sk-your-actual-api-key"

# Add Azure OpenAI key
az keyvault secret set --vault-name MyKeyVault --name AZURE_OPENAI_KEY --value "your-azure-openai-key"

# Add any other secrets you need
az keyvault secret set --vault-name MyKeyVault --name DATABASE_PASSWORD --value "your-db-password"
```

### 3. Set Up Authentication to Azure Key Vault

You have two options for authenticating to Azure Key Vault:

#### Option A: Service Principal (works anywhere)

Create a service principal with permissions to access your Key Vault:

```bash
# Create a service principal and store the credentials
az ad sp create-for-rbac --name "MyAppServicePrincipal" --skip-assignment

# Note the appId (client_id), password (client_secret), and tenant values from the output

# Grant the service principal access to your Key Vault secrets
az keyvault set-policy --name MyKeyVault --spn [appId-value] --secret-permissions get list
```

#### Option B: Managed Identity (Azure deployments only)

If your app is deployed to Azure, you can use managed identity for seamless authentication:

1. Enable managed identity on your Azure service (App Service, Azure Functions, etc.)
2. Grant the managed identity permission to access Key Vault secrets:

```bash
# Get the principal ID of your managed identity
az webapp identity show --name MyWebApp --resource-group MyResourceGroup

# Grant access to Key Vault
az keyvault set-policy --name MyKeyVault --object-id [principal-id] --secret-permissions get list
```

### 4. Configure Streamlit Secrets

Create a `.streamlit/secrets.toml` file with Azure Key Vault connection details:

```toml
# Azure Key Vault configuration
[key_vault]
KEY_VAULT_ENABLED = true
KEY_VAULT_URL = "https://your-vault-name.vault.azure.net/"

# Choose one authentication method:

# Option A: Service Principal Authentication
USE_MANAGED_IDENTITY = false
CLIENT_ID = "your-service-principal-client-id"
CLIENT_SECRET = "your-service-principal-client-secret"
TENANT_ID = "your-azure-tenant-id"

# Option B: Managed Identity Authentication
# USE_MANAGED_IDENTITY = true
```

**Important:** Never commit your actual `.streamlit/secrets.toml` file to source control! Add it to `.gitignore`.

### 5. Local Development Configuration

For local development, you have two options:

#### Option 1: Use Azure Key Vault locally

Use the same Azure Key Vault configuration as production, but with service principal authentication (Option A above).

#### Option 2: Use Streamlit secrets as fallbacks

Add local development fallback values to your `.streamlit/secrets.toml`:

```toml
# Local development fallbacks (only used if Key Vault is not accessible)
OPENAI_API_KEY = "your-local-development-key"
```

## How It Works

1. When your app starts, it first loads settings from environment variables
2. Then it merges in values from Streamlit secrets
3. If Key Vault is enabled, it connects to Azure Key Vault using the connection details
4. Secrets retrieved from Key Vault are added to environment variables
5. The settings are recreated to pick up the values from Key Vault
6. For improved performance, secrets can be cached in memory and refreshed periodically

## Priority Order

Secrets are loaded with the following priority (highest to lowest):

1. Explicit environment variables (set directly in the environment)
2. Secrets from Azure Key Vault
3. Values from Streamlit secrets
4. Default values defined in Pydantic settings

This ensures that you can override any value when needed without changing code.

## Deployment to Streamlit Cloud

When deploying to Streamlit Cloud:

1. Copy your `.streamlit/secrets.toml` file to the "Secrets" section in the app settings
2. Make sure to include the Azure Key Vault configuration
3. The app will use these settings to connect to Azure Key Vault

## Deployment to Azure

When deploying to Azure:

1. Enable managed identity for your Azure service
2. Grant the managed identity access to your Key Vault
3. Configure the following app settings (environment variables):
   - `key_vault__KEY_VAULT_ENABLED=true`
   - `key_vault__KEY_VAULT_URL=https://your-vault-name.vault.azure.net/`
   - `key_vault__USE_MANAGED_IDENTITY=true`

## Troubleshooting

If you experience issues with secret management:

1. Check that Key Vault is accessible from your environment
2. Verify that the authentication credentials (service principal or managed identity) have appropriate permissions
3. Enable debug mode to see which secrets are being loaded
4. Check the application logs for error messages 