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

# Container Instance Outputs
output "container_instance_name" {
  description = "The name of the container instance"
  value       = azurerm_container_group.app.name
}

output "container_instance_fqdn" {
  description = "The FQDN of the container instance"
  value       = azurerm_container_group.app.fqdn
}

output "container_instance_ip" {
  description = "The IP address of the container instance"
  value       = azurerm_container_group.app.ip_address
}

# Application URL
output "app_url" {
  description = "The URL to access the application"
  value       = "http://${azurerm_container_group.app.fqdn}:8501"
} 