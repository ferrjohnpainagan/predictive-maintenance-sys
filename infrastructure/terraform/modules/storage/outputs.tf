output "s3_bucket_name" {
  description = "S3 bucket name for ML data"
  value       = aws_s3_bucket.ml_data.id
}

output "s3_bucket_arn" {
  description = "S3 bucket ARN"
  value       = aws_s3_bucket.ml_data.arn
}

output "s3_bucket_region" {
  description = "S3 bucket region"
  value       = aws_s3_bucket.ml_data.region
}
