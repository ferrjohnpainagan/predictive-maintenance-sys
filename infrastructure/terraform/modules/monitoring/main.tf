# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/ec2/${var.project}-${var.environment}-api-gateway"
  retention_in_days = 7  # Free tier optimization
}

resource "aws_cloudwatch_log_group" "ml_service" {
  name              = "/aws/ec2/${var.project}-${var.environment}-ml-service"
  retention_in_days = 7  # Free tier optimization
}

# CloudWatch Metric Alarms
resource "aws_cloudwatch_metric_alarm" "api_gateway_cpu_high" {
  alarm_name          = "${var.project}-${var.environment}-api-gateway-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 80

  dimensions = {
    InstanceId = var.instance_ids[0]
  }

  alarm_description = "CPU utilization is too high for API Gateway"
}

resource "aws_cloudwatch_metric_alarm" "ml_service_cpu_high" {
  alarm_name          = "${var.project}-${var.environment}-ml-service-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 80

  dimensions = {
    InstanceId = var.instance_ids[1]
  }

  alarm_description = "CPU utilization is too high for ML Service"
}

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.project}-${var.environment}-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/EC2", "CPUUtilization", "InstanceId", var.instance_ids[0], { "label": "API Gateway CPU" }],
            ["AWS/EC2", "CPUUtilization", "InstanceId", var.instance_ids[1], { "label": "ML Service CPU" }]
          ]
          period = 300
          stat   = "Average"
          region = "ap-southeast-1"
          title  = "EC2 CPU Utilization"
        }
      }
    ]
  })
}
