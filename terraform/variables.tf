############################
# Project Configuration #
############################
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "flatbot"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

############################
# Azure Configuration     #
############################
variable "location" {
  description = "Azure region to deploy resources"
  type        = string
  default     = "westus2"  # Western US region for lower latency
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default = {
    Environment = "dev"
    Project     = "flat-chatbot"
    ManagedBy   = "terraform"
    CostCenter  = "minimized"
  }
}

############################
# Application Configuration #
############################ 