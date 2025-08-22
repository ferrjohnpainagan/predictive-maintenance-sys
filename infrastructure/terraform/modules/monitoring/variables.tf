variable "project" {
  description = "Project name"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "instance_ids" {
  description = "EC2 instance IDs for monitoring"
  type        = list(string)
}

variable "security_group_id" {
  description = "Security group ID"
  type        = string
}
