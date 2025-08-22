output "log_group_names" {
  description = "CloudWatch log group names"
  value = [
    aws_cloudwatch_log_group.api_gateway.name,
    aws_cloudwatch_log_group.ml_service.name
  ]
}

output "dashboard_name" {
  description = "CloudWatch dashboard name"
  value       = aws_cloudwatch_dashboard.main.dashboard_name
}

output "alarm_names" {
  description = "CloudWatch alarm names"
  value = [
    aws_cloudwatch_metric_alarm.api_gateway_cpu_high.alarm_name,
    aws_cloudwatch_metric_alarm.ml_service_cpu_high.alarm_name
  ]
}
