output "api_gateway_id" {
  description = "API Gateway instance ID"
  value       = aws_instance.api_gateway.id
}

output "ml_service_id" {
  description = "ML Service instance ID"
  value       = aws_instance.ml_service.id
}

output "instance_ids" {
  description = "All EC2 instance IDs"
  value       = [aws_instance.api_gateway.id, aws_instance.ml_service.id]
}

output "api_gateway_public_ip" {
  description = "API Gateway public IP"
  value       = aws_instance.api_gateway.public_ip
}

output "ml_service_public_ip" {
  description = "ML Service public IP"
  value       = aws_instance.ml_service.public_ip
}

output "api_gateway_private_ip" {
  description = "API Gateway private IP"
  value       = aws_instance.api_gateway.private_ip
}

output "ml_service_private_ip" {
  description = "ML Service private IP"
  value       = aws_instance.ml_service.private_ip
}
