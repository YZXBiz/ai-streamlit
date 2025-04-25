terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
  
  # Using local state for simplicity
  # For production, consider using remote state with Azure Storage
}

# Resource group for all resources
resource "azurerm_resource_group" "this" {
  name     = "rg-${var.project_name}-${var.environment}"
  location = var.location
  tags     = var.tags
}

# Azure Container Registry - using Basic SKU (cheapest option)
resource "azurerm_container_registry" "acr" {
  name                = "acr${var.project_name}${var.environment}"
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  sku                 = "Basic"
  admin_enabled       = true
  tags                = var.tags
}

# Azure Container Instances - optimized for minimal resource usage
resource "azurerm_container_group" "app" {
  name                = "aci-${var.project_name}-${var.environment}"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  ip_address_type     = "Public"
  dns_name_label      = "${var.project_name}-${var.environment}"
  os_type             = "Linux"
  
  container {
    name   = var.project_name
    image  = "${azurerm_container_registry.acr.login_server}/${var.project_name}:latest"
    cpu    = "0.5"
    memory = "1.0"
    
    ports {
      port     = 8501
      protocol = "TCP"
    }
    
    environment_variables = {
      "OPENAI_API_KEY" = var.openai_api_key
    }
  }
  
  image_registry_credential {
    server   = azurerm_container_registry.acr.login_server
    username = azurerm_container_registry.acr.admin_username
    password = azurerm_container_registry.acr.admin_password
  }
  
  tags = var.tags
} 