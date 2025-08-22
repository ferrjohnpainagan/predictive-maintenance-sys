# Project Configuration
variable "project" {
  description = "Project name"
  type        = string
  default     = "predictive-maintenance"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "free-tier"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-southeast-1"
}

# Supabase Configuration
variable "supabase_url" {
  description = "Supabase project URL"
  type        = string
}

variable "supabase_anon_key" {
  description = "Supabase anonymous key"
  type        = string
  sensitive   = true
}

variable "supabase_service_role_key" {
  description = "Supabase service role key"
  type        = string
  sensitive   = true
}

# SSH Key Configuration
variable "ssh_key_name" {
  description = "Name of the SSH key pair"
  type        = string
  default     = "predictive-maintenance-key"
}

# Tags
variable "common_tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project     = "predictive-maintenance"
    Environment = "free-tier"
    ManagedBy   = "terraform"
    Purpose     = "predictive-maintenance-system"
  }
}
