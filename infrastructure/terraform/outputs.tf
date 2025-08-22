# Networking Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = module.networking.vpc_id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = module.networking.public_subnet_ids
}

# Compute Outputs
output "api_gateway_public_ip" {
  description = "API Gateway public IP"
  value       = module.compute.api_gateway_public_ip
}

output "ml_service_public_ip" {
  description = "ML Service public IP"
  value       = module.compute.ml_service_public_ip
}

output "instance_ids" {
  description = "EC2 instance IDs"
  value       = module.compute.instance_ids
}

# Storage Outputs
output "s3_bucket_name" {
  description = "S3 bucket name for ML data"
  value       = module.storage.s3_bucket_name
}

# Monitoring Outputs
output "cloudwatch_log_groups" {
  description = "CloudWatch log group names"
  value       = module.monitoring.log_group_names
}

# Connection Information
output "connection_info" {
  description = "Connection information for services"
  value = {
    api_gateway = "http://${module.compute.api_gateway_public_ip}:3000"
    ml_service  = "http://${module.compute.ml_service_public_ip}:8000"
    ssh_command = "ssh -i ~/.ssh/${var.ssh_key_name} ec2-user@${module.compute.api_gateway_public_ip}"
  }
}
