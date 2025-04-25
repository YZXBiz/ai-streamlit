############################
# Resource Information     #
############################

# Resource Group
output "resource_group_name" {
  description = "The name of the resource group"
  value       = azurerm_resource_group.this.name
}

# Container Registry Outputs
output "acr_name" {
  description = "The Azure Container Registry name"
  value       = azurerm_container_registry.acr.name
}

output "acr_login_server" {
  description = "The Azure Container Registry login server"
  value       = azurerm_container_registry.acr.login_server
}

# Container Registry Credentials
output "acr_admin_username" {
  description = "The admin username for the Azure Container Registry"
  value       = azurerm_container_registry.acr.admin_username
  sensitive   = true
}

output "acr_admin_password" {
  description = "The admin password for the Azure Container Registry"
  value       = azurerm_container_registry.acr.admin_password
  sensitive   = true
}

# Note: Container Instance outputs removed
# GitHub Actions will manage container deployment and provide the app URL 