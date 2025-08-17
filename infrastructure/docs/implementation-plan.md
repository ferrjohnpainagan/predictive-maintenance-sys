# Infrastructure as Code Implementation Plan - Predictive Maintenance System

## Project Overview
Implementation plan for the complete infrastructure setup using Infrastructure as Code (IaC) principles. This covers AWS resources, Supabase configuration, Vercel deployment, and the entire network architecture required for the predictive maintenance system.

## Technology Stack
- **IaC Tools**: Terraform (primary), AWS CDK (alternative)
- **Cloud Providers**: AWS, Supabase, Vercel
- **CI/CD**: GitHub Actions, Terraform Cloud
- **Secret Management**: AWS Secrets Manager, GitHub Secrets
- **Monitoring**: AWS CloudWatch, Datadog

---

## Phase 1: Foundation and Prerequisites
**Duration**: 2-3 days  
**Priority**: Critical

### Objectives
- Set up IaC tooling
- Configure state management
- Establish account structure

### Tasks
1. **Terraform Setup**
   ```hcl
   # terraform/versions.tf
   terraform {
     required_version = ">= 1.5.0"
     
     required_providers {
       aws = {
         source  = "hashicorp/aws"
         version = "~> 5.0"
       }
       supabase = {
         source  = "supabase/supabase"
         version = "~> 1.0"
       }
       vercel = {
         source  = "vercel/vercel"
         version = "~> 0.15"
       }
     }
     
     backend "s3" {
       bucket         = "predictive-maintenance-terraform-state"
       key            = "infrastructure/terraform.tfstate"
       region         = "us-east-1"
       dynamodb_table = "terraform-state-lock"
       encrypt        = true
     }
   }
   ```

2. **Project Structure**
   ```
   infrastructure/
   ├── terraform/
   │   ├── environments/
   │   │   ├── dev/
   │   │   ├── staging/
   │   │   └── production/
   │   ├── modules/
   │   │   ├── networking/
   │   │   ├── compute/
   │   │   ├── database/
   │   │   ├── storage/
   │   │   ├── monitoring/
   │   │   └── security/
   │   ├── global/
   │   └── scripts/
   ├── kubernetes/
   │   ├── manifests/
   │   └── helm/
   └── docs/
   ```

3. **State Management Setup**
   ```bash
   # scripts/setup-backend.sh
   #!/bin/bash
   
   # Create S3 bucket for Terraform state
   aws s3api create-bucket \
     --bucket predictive-maintenance-terraform-state \
     --region us-east-1 \
     --acl private
   
   # Enable versioning
   aws s3api put-bucket-versioning \
     --bucket predictive-maintenance-terraform-state \
     --versioning-configuration Status=Enabled
   
   # Enable encryption
   aws s3api put-bucket-encryption \
     --bucket predictive-maintenance-terraform-state \
     --server-side-encryption-configuration '{
       "Rules": [{
         "ApplyServerSideEncryptionByDefault": {
           "SSEAlgorithm": "AES256"
         }
       }]
     }'
   
   # Create DynamoDB table for state locking
   aws dynamodb create-table \
     --table-name terraform-state-lock \
     --attribute-definitions AttributeName=LockID,AttributeType=S \
     --key-schema AttributeName=LockID,KeyType=HASH \
     --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
   ```

4. **Provider Configuration**
   ```hcl
   # terraform/providers.tf
   provider "aws" {
     region = var.aws_region
     
     default_tags {
       tags = {
         Project     = "PredictiveMaintenance"
         Environment = var.environment
         ManagedBy   = "Terraform"
         CostCenter  = "Engineering"
       }
     }
   }
   
   provider "supabase" {
     api_key = var.supabase_api_key
   }
   
   provider "vercel" {
     api_token = var.vercel_api_token
   }
   ```

### Deliverables
- Terraform configuration initialized
- State backend configured
- Provider authentication set up
- Project structure created

### Testing Checklist
- [ ] Terraform init successful
- [ ] State backend accessible
- [ ] Provider authentication works
- [ ] Directory structure created

---

## Phase 2: Network Infrastructure
**Duration**: 3-4 days  
**Priority**: Critical

### Objectives
- Create VPC architecture
- Configure security groups
- Set up networking components

### Tasks
1. **VPC Module**
   ```hcl
   # modules/networking/vpc.tf
   resource "aws_vpc" "main" {
     cidr_block           = var.vpc_cidr
     enable_dns_hostnames = true
     enable_dns_support   = true
     
     tags = {
       Name = "${var.project}-${var.environment}-vpc"
     }
   }
   
   # Public Subnets
   resource "aws_subnet" "public" {
     count                   = length(var.availability_zones)
     vpc_id                  = aws_vpc.main.id
     cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
     availability_zone       = var.availability_zones[count.index]
     map_public_ip_on_launch = true
     
     tags = {
       Name = "${var.project}-${var.environment}-public-${count.index + 1}"
       Type = "Public"
     }
   }
   
   # Private Subnets
   resource "aws_subnet" "private" {
     count             = length(var.availability_zones)
     vpc_id            = aws_vpc.main.id
     cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
     availability_zone = var.availability_zones[count.index]
     
     tags = {
       Name = "${var.project}-${var.environment}-private-${count.index + 1}"
       Type = "Private"
     }
   }
   
   # Internet Gateway
   resource "aws_internet_gateway" "main" {
     vpc_id = aws_vpc.main.id
     
     tags = {
       Name = "${var.project}-${var.environment}-igw"
     }
   }
   
   # NAT Gateways
   resource "aws_eip" "nat" {
     count  = var.enable_nat_gateway ? length(var.availability_zones) : 0
     domain = "vpc"
     
     tags = {
       Name = "${var.project}-${var.environment}-nat-eip-${count.index + 1}"
     }
   }
   
   resource "aws_nat_gateway" "main" {
     count         = var.enable_nat_gateway ? length(var.availability_zones) : 0
     allocation_id = aws_eip.nat[count.index].id
     subnet_id     = aws_subnet.public[count.index].id
     
     tags = {
       Name = "${var.project}-${var.environment}-nat-${count.index + 1}"
     }
   }
   ```

2. **Security Groups**
   ```hcl
   # modules/networking/security_groups.tf
   
   # ALB Security Group
   resource "aws_security_group" "alb" {
     name_prefix = "${var.project}-${var.environment}-alb-"
     vpc_id      = aws_vpc.main.id
     
     ingress {
       from_port   = 80
       to_port     = 80
       protocol    = "tcp"
       cidr_blocks = ["0.0.0.0/0"]
     }
     
     ingress {
       from_port   = 443
       to_port     = 443
       protocol    = "tcp"
       cidr_blocks = ["0.0.0.0/0"]
     }
     
     egress {
       from_port   = 0
       to_port     = 0
       protocol    = "-1"
       cidr_blocks = ["0.0.0.0/0"]
     }
     
     tags = {
       Name = "${var.project}-${var.environment}-alb-sg"
     }
   }
   
   # ECS/Fargate Security Group
   resource "aws_security_group" "ecs" {
     name_prefix = "${var.project}-${var.environment}-ecs-"
     vpc_id      = aws_vpc.main.id
     
     ingress {
       from_port       = 3000
       to_port         = 3000
       protocol        = "tcp"
       security_groups = [aws_security_group.alb.id]
     }
     
     ingress {
       from_port = 50051
       to_port   = 50051
       protocol  = "tcp"
       self      = true
     }
     
     egress {
       from_port   = 0
       to_port     = 0
       protocol    = "-1"
       cidr_blocks = ["0.0.0.0/0"]
     }
     
     tags = {
       Name = "${var.project}-${var.environment}-ecs-sg"
     }
   }
   
   # SageMaker Security Group
   resource "aws_security_group" "sagemaker" {
     name_prefix = "${var.project}-${var.environment}-sagemaker-"
     vpc_id      = aws_vpc.main.id
     
     ingress {
       from_port = 443
       to_port   = 443
       protocol  = "tcp"
       self      = true
     }
     
     egress {
       from_port   = 0
       to_port     = 0
       protocol    = "-1"
       cidr_blocks = ["0.0.0.0/0"]
     }
     
     tags = {
       Name = "${var.project}-${var.environment}-sagemaker-sg"
     }
   }
   ```

3. **Route Tables**
   ```hcl
   # modules/networking/routes.tf
   
   # Public Route Table
   resource "aws_route_table" "public" {
     vpc_id = aws_vpc.main.id
     
     route {
       cidr_block = "0.0.0.0/0"
       gateway_id = aws_internet_gateway.main.id
     }
     
     tags = {
       Name = "${var.project}-${var.environment}-public-rt"
     }
   }
   
   resource "aws_route_table_association" "public" {
     count          = length(aws_subnet.public)
     subnet_id      = aws_subnet.public[count.index].id
     route_table_id = aws_route_table.public.id
   }
   
   # Private Route Tables
   resource "aws_route_table" "private" {
     count  = length(var.availability_zones)
     vpc_id = aws_vpc.main.id
     
     route {
       cidr_block     = "0.0.0.0/0"
       nat_gateway_id = var.enable_nat_gateway ? aws_nat_gateway.main[count.index].id : null
     }
     
     tags = {
       Name = "${var.project}-${var.environment}-private-rt-${count.index + 1}"
     }
   }
   ```

### Deliverables
- VPC with public/private subnets
- Security groups configured
- NAT gateways for private subnets
- Route tables set up

### Testing Checklist
- [ ] VPC created successfully
- [ ] Subnets in multiple AZs
- [ ] Security groups allow required traffic
- [ ] Internet connectivity verified

---

## Phase 3: Compute Infrastructure
**Duration**: 4-5 days  
**Priority**: High

### Objectives
- Set up ECS/Fargate clusters
- Configure auto-scaling
- Deploy load balancers

### Tasks
1. **ECS Cluster**
   ```hcl
   # modules/compute/ecs.tf
   resource "aws_ecs_cluster" "main" {
     name = "${var.project}-${var.environment}-cluster"
     
     setting {
       name  = "containerInsights"
       value = "enabled"
     }
     
     configuration {
       execute_command_configuration {
         logging = "OVERRIDE"
         
         log_configuration {
           cloud_watch_encryption_enabled = true
           cloud_watch_log_group_name     = aws_cloudwatch_log_group.ecs.name
         }
       }
     }
   }
   
   # ECS Task Definition - API Gateway
   resource "aws_ecs_task_definition" "api_gateway" {
     family                   = "${var.project}-api-gateway"
     network_mode             = "awsvpc"
     requires_compatibilities = ["FARGATE"]
     cpu                      = "1024"
     memory                   = "2048"
     execution_role_arn       = aws_iam_role.ecs_execution.arn
     task_role_arn           = aws_iam_role.ecs_task.arn
     
     container_definitions = jsonencode([
       {
         name  = "api-gateway"
         image = "${aws_ecr_repository.api_gateway.repository_url}:latest"
         
         portMappings = [
           {
             containerPort = 3000
             protocol      = "tcp"
           }
         ]
         
         environment = [
           {
             name  = "NODE_ENV"
             value = var.environment
           },
           {
             name  = "SUPABASE_URL"
             value = var.supabase_url
           },
           {
             name  = "ML_SERVICE_HOST"
             value = "localhost"
           },
           {
             name  = "ML_SERVICE_PORT"
             value = "50051"
           }
         ]
         
         secrets = [
           {
             name      = "SUPABASE_SERVICE_KEY"
             valueFrom = aws_secretsmanager_secret_version.supabase_key.arn
           }
         ]
         
         logConfiguration = {
           logDriver = "awslogs"
           options = {
             "awslogs-group"         = aws_cloudwatch_log_group.api_gateway.name
             "awslogs-region"        = var.aws_region
             "awslogs-stream-prefix" = "ecs"
           }
         }
       }
     ])
   }
   
   # ECS Service
   resource "aws_ecs_service" "api_gateway" {
     name            = "${var.project}-api-gateway"
     cluster         = aws_ecs_cluster.main.id
     task_definition = aws_ecs_task_definition.api_gateway.arn
     desired_count   = var.api_gateway_count
     launch_type     = "FARGATE"
     
     network_configuration {
       subnets          = aws_subnet.private[*].id
       security_groups  = [aws_security_group.ecs.id]
       assign_public_ip = false
     }
     
     load_balancer {
       target_group_arn = aws_lb_target_group.api_gateway.arn
       container_name   = "api-gateway"
       container_port   = 3000
     }
     
     deployment_configuration {
       maximum_percent         = 200
       minimum_healthy_percent = 100
     }
     
     depends_on = [aws_lb_listener.main]
   }
   ```

2. **Application Load Balancer**
   ```hcl
   # modules/compute/alb.tf
   resource "aws_lb" "main" {
     name               = "${var.project}-${var.environment}-alb"
     internal           = false
     load_balancer_type = "application"
     security_groups    = [aws_security_group.alb.id]
     subnets           = aws_subnet.public[*].id
     
     enable_deletion_protection = var.environment == "production"
     enable_http2              = true
     enable_cross_zone_load_balancing = true
     
     access_logs {
       bucket  = aws_s3_bucket.alb_logs.bucket
       enabled = true
     }
   }
   
   resource "aws_lb_target_group" "api_gateway" {
     name        = "${var.project}-api-gateway-tg"
     port        = 3000
     protocol    = "HTTP"
     vpc_id      = aws_vpc.main.id
     target_type = "ip"
     
     health_check {
       enabled             = true
       healthy_threshold   = 2
       unhealthy_threshold = 2
       timeout             = 5
       interval            = 30
       path                = "/health"
       matcher             = "200"
     }
     
     deregistration_delay = 30
   }
   
   resource "aws_lb_listener" "main" {
     load_balancer_arn = aws_lb.main.arn
     port              = "443"
     protocol          = "HTTPS"
     ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
     certificate_arn   = aws_acm_certificate.main.arn
     
     default_action {
       type             = "forward"
       target_group_arn = aws_lb_target_group.api_gateway.arn
     }
   }
   
   resource "aws_lb_listener" "redirect" {
     load_balancer_arn = aws_lb.main.arn
     port              = "80"
     protocol          = "HTTP"
     
     default_action {
       type = "redirect"
       
       redirect {
         port        = "443"
         protocol    = "HTTPS"
         status_code = "HTTP_301"
       }
     }
   }
   ```

3. **Auto Scaling**
   ```hcl
   # modules/compute/autoscaling.tf
   resource "aws_appautoscaling_target" "ecs" {
     max_capacity       = var.max_capacity
     min_capacity       = var.min_capacity
     resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.api_gateway.name}"
     scalable_dimension = "ecs:service:DesiredCount"
     service_namespace  = "ecs"
   }
   
   resource "aws_appautoscaling_policy" "cpu" {
     name               = "${var.project}-cpu-scaling"
     policy_type        = "TargetTrackingScaling"
     resource_id        = aws_appautoscaling_target.ecs.resource_id
     scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
     service_namespace  = aws_appautoscaling_target.ecs.service_namespace
     
     target_tracking_scaling_policy_configuration {
       predefined_metric_specification {
         predefined_metric_type = "ECSServiceAverageCPUUtilization"
       }
       
       target_value = 70.0
     }
   }
   
   resource "aws_appautoscaling_policy" "memory" {
     name               = "${var.project}-memory-scaling"
     policy_type        = "TargetTrackingScaling"
     resource_id        = aws_appautoscaling_target.ecs.resource_id
     scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
     service_namespace  = aws_appautoscaling_target.ecs.service_namespace
     
     target_tracking_scaling_policy_configuration {
       predefined_metric_specification {
         predefined_metric_type = "ECSServiceAverageMemoryUtilization"
       }
       
       target_value = 80.0
     }
   }
   ```

### Deliverables
- ECS cluster configured
- Task definitions created
- Load balancer deployed
- Auto-scaling policies set

### Testing Checklist
- [ ] ECS cluster healthy
- [ ] Tasks running successfully
- [ ] Load balancer accessible
- [ ] Auto-scaling triggers work

---

## Phase 4: Storage and Database
**Duration**: 3-4 days  
**Priority**: High

### Objectives
- Configure S3 buckets
- Set up ECR repositories
- Configure Supabase integration

### Tasks
1. **S3 Buckets**
   ```hcl
   # modules/storage/s3.tf
   
   # ML Models and Data Storage
   resource "aws_s3_bucket" "ml_data" {
     bucket = "${var.project}-${var.environment}-ml-data"
   }
   
   resource "aws_s3_bucket_versioning" "ml_data" {
     bucket = aws_s3_bucket.ml_data.id
     
     versioning_configuration {
       status = "Enabled"
     }
   }
   
   resource "aws_s3_bucket_encryption" "ml_data" {
     bucket = aws_s3_bucket.ml_data.id
     
     rule {
       apply_server_side_encryption_by_default {
         sse_algorithm = "AES256"
       }
     }
   }
   
   resource "aws_s3_bucket_lifecycle_configuration" "ml_data" {
     bucket = aws_s3_bucket.ml_data.id
     
     rule {
       id     = "archive-old-models"
       status = "Enabled"
       
       transition {
         days          = 30
         storage_class = "STANDARD_IA"
       }
       
       transition {
         days          = 90
         storage_class = "GLACIER"
       }
     }
   }
   
   # ALB Logs Bucket
   resource "aws_s3_bucket" "alb_logs" {
     bucket = "${var.project}-${var.environment}-alb-logs"
   }
   
   resource "aws_s3_bucket_policy" "alb_logs" {
     bucket = aws_s3_bucket.alb_logs.id
     
     policy = jsonencode({
       Version = "2012-10-17"
       Statement = [
         {
           Effect = "Allow"
           Principal = {
             AWS = "arn:aws:iam::${data.aws_elb_service_account.main.id}:root"
           }
           Action   = "s3:PutObject"
           Resource = "${aws_s3_bucket.alb_logs.arn}/*"
         }
       ]
     })
   }
   ```

2. **ECR Repositories**
   ```hcl
   # modules/storage/ecr.tf
   
   # API Gateway Repository
   resource "aws_ecr_repository" "api_gateway" {
     name                 = "${var.project}/api-gateway"
     image_tag_mutability = "MUTABLE"
     
     image_scanning_configuration {
       scan_on_push = true
     }
     
     encryption_configuration {
       encryption_type = "AES256"
     }
   }
   
   resource "aws_ecr_lifecycle_policy" "api_gateway" {
     repository = aws_ecr_repository.api_gateway.name
     
     policy = jsonencode({
       rules = [
         {
           rulePriority = 1
           description  = "Keep last 10 images"
           selection = {
             tagStatus     = "tagged"
             tagPrefixList = ["v"]
             countType     = "imageCountMoreThan"
             countNumber   = 10
           }
           action = {
             type = "expire"
           }
         }
       ]
     })
   }
   
   # ML Service Repository
   resource "aws_ecr_repository" "ml_service" {
     name                 = "${var.project}/ml-service"
     image_tag_mutability = "MUTABLE"
     
     image_scanning_configuration {
       scan_on_push = true
     }
   }
   ```

3. **Supabase Configuration**
   ```hcl
   # modules/database/supabase.tf
   
   # Note: Supabase provider is limited, most config done via API
   resource "null_resource" "supabase_setup" {
     provisioner "local-exec" {
       command = <<-EOT
         curl -X POST https://api.supabase.com/v1/projects \
           -H "Authorization: Bearer ${var.supabase_api_key}" \
           -H "Content-Type: application/json" \
           -d '{
             "name": "${var.project}-${var.environment}",
             "region": "us-east-1",
             "plan": "pro",
             "db_pass": "${random_password.supabase_db.result}",
             "organization_id": "${var.supabase_org_id}"
           }'
       EOT
     }
   }
   
   # Store Supabase credentials in Secrets Manager
   resource "aws_secretsmanager_secret" "supabase" {
     name = "${var.project}-${var.environment}-supabase"
   }
   
   resource "aws_secretsmanager_secret_version" "supabase" {
     secret_id = aws_secretsmanager_secret.supabase.id
     
     secret_string = jsonencode({
       url         = var.supabase_url
       anon_key    = var.supabase_anon_key
       service_key = var.supabase_service_key
       db_password = random_password.supabase_db.result
     })
   }
   ```

### Deliverables
- S3 buckets configured
- ECR repositories created
- Supabase project initialized
- Secrets stored securely

### Testing Checklist
- [ ] S3 buckets accessible
- [ ] ECR repositories created
- [ ] Supabase connection works
- [ ] Secrets retrievable

---

## Phase 5: Security and IAM
**Duration**: 3-4 days  
**Priority**: Critical

### Objectives
- Configure IAM roles and policies
- Set up secrets management
- Implement security best practices

### Tasks
1. **IAM Roles**
   ```hcl
   # modules/security/iam.tf
   
   # ECS Task Execution Role
   resource "aws_iam_role" "ecs_execution" {
     name = "${var.project}-${var.environment}-ecs-execution"
     
     assume_role_policy = jsonencode({
       Version = "2012-10-17"
       Statement = [
         {
           Action = "sts:AssumeRole"
           Effect = "Allow"
           Principal = {
             Service = "ecs-tasks.amazonaws.com"
           }
         }
       ]
     })
   }
   
   resource "aws_iam_role_policy_attachment" "ecs_execution" {
     role       = aws_iam_role.ecs_execution.name
     policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
   }
   
   resource "aws_iam_role_policy" "ecs_execution_secrets" {
     name = "secrets-access"
     role = aws_iam_role.ecs_execution.id
     
     policy = jsonencode({
       Version = "2012-10-17"
       Statement = [
         {
           Effect = "Allow"
           Action = [
             "secretsmanager:GetSecretValue"
           ]
           Resource = [
             aws_secretsmanager_secret.supabase.arn
           ]
         }
       ]
     })
   }
   
   # ECS Task Role
   resource "aws_iam_role" "ecs_task" {
     name = "${var.project}-${var.environment}-ecs-task"
     
     assume_role_policy = jsonencode({
       Version = "2012-10-17"
       Statement = [
         {
           Action = "sts:AssumeRole"
           Effect = "Allow"
           Principal = {
             Service = "ecs-tasks.amazonaws.com"
           }
         }
       ]
     })
   }
   
   resource "aws_iam_role_policy" "ecs_task" {
     name = "task-permissions"
     role = aws_iam_role.ecs_task.id
     
     policy = jsonencode({
       Version = "2012-10-17"
       Statement = [
         {
           Effect = "Allow"
           Action = [
             "s3:GetObject",
             "s3:PutObject"
           ]
           Resource = [
             "${aws_s3_bucket.ml_data.arn}/*"
           ]
         },
         {
           Effect = "Allow"
           Action = [
             "sagemaker:InvokeEndpoint"
           ]
           Resource = [
             "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:endpoint/*"
           ]
         }
       ]
     })
   }
   
   # SageMaker Execution Role
   resource "aws_iam_role" "sagemaker" {
     name = "${var.project}-${var.environment}-sagemaker"
     
     assume_role_policy = jsonencode({
       Version = "2012-10-17"
       Statement = [
         {
           Action = "sts:AssumeRole"
           Effect = "Allow"
           Principal = {
             Service = "sagemaker.amazonaws.com"
           }
         }
       ]
     })
   }
   
   resource "aws_iam_role_policy_attachment" "sagemaker_full" {
     role       = aws_iam_role.sagemaker.name
     policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
   }
   ```

2. **KMS Encryption**
   ```hcl
   # modules/security/kms.tf
   resource "aws_kms_key" "main" {
     description             = "${var.project}-${var.environment} encryption key"
     deletion_window_in_days = 10
     enable_key_rotation     = true
   }
   
   resource "aws_kms_alias" "main" {
     name          = "alias/${var.project}-${var.environment}"
     target_key_id = aws_kms_key.main.key_id
   }
   
   resource "aws_kms_key_policy" "main" {
     key_id = aws_kms_key.main.id
     
     policy = jsonencode({
       Version = "2012-10-17"
       Statement = [
         {
           Sid    = "Enable IAM User Permissions"
           Effect = "Allow"
           Principal = {
             AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
           }
           Action   = "kms:*"
           Resource = "*"
         },
         {
           Sid    = "Allow services to use the key"
           Effect = "Allow"
           Principal = {
             Service = [
               "ecs.amazonaws.com",
               "s3.amazonaws.com",
               "secretsmanager.amazonaws.com"
             ]
           }
           Action = [
             "kms:Decrypt",
             "kms:GenerateDataKey"
           ]
           Resource = "*"
         }
       ]
     })
   }
   ```

3. **WAF Configuration**
   ```hcl
   # modules/security/waf.tf
   resource "aws_wafv2_web_acl" "main" {
     name  = "${var.project}-${var.environment}-waf"
     scope = "REGIONAL"
     
     default_action {
       allow {}
     }
     
     rule {
       name     = "RateLimitRule"
       priority = 1
       
       statement {
         rate_based_statement {
           limit              = 2000
           aggregate_key_type = "IP"
         }
       }
       
       action {
         block {}
       }
       
       visibility_config {
         cloudwatch_metrics_enabled = true
         metric_name               = "RateLimitRule"
         sampled_requests_enabled   = true
       }
     }
     
     rule {
       name     = "AWSManagedRulesCommonRuleSet"
       priority = 2
       
       override_action {
         none {}
       }
       
       statement {
         managed_rule_group_statement {
           name        = "AWSManagedRulesCommonRuleSet"
           vendor_name = "AWS"
         }
       }
       
       visibility_config {
         cloudwatch_metrics_enabled = true
         metric_name               = "CommonRuleSet"
         sampled_requests_enabled   = true
       }
     }
   }
   
   resource "aws_wafv2_web_acl_association" "alb" {
     resource_arn = aws_lb.main.arn
     web_acl_arn  = aws_wafv2_web_acl.main.arn
   }
   ```

### Deliverables
- IAM roles and policies configured
- KMS encryption enabled
- WAF rules implemented
- Security best practices applied

### Testing Checklist
- [ ] IAM roles have correct permissions
- [ ] Encryption working
- [ ] WAF rules active
- [ ] Security scan passed

---

## Phase 6: Monitoring and Observability
**Duration**: 3-4 days  
**Priority**: High

### Objectives
- Set up CloudWatch monitoring
- Configure alarms and notifications
- Implement logging strategy

### Tasks
1. **CloudWatch Configuration**
   ```hcl
   # modules/monitoring/cloudwatch.tf
   
   # Log Groups
   resource "aws_cloudwatch_log_group" "ecs" {
     name              = "/ecs/${var.project}-${var.environment}"
     retention_in_days = var.log_retention_days
     
     kms_key_id = aws_kms_key.main.arn
   }
   
   resource "aws_cloudwatch_log_group" "api_gateway" {
     name              = "/aws/ecs/${var.project}-api-gateway"
     retention_in_days = var.log_retention_days
   }
   
   resource "aws_cloudwatch_log_group" "ml_service" {
     name              = "/aws/ecs/${var.project}-ml-service"
     retention_in_days = var.log_retention_days
   }
   
   # CloudWatch Dashboard
   resource "aws_cloudwatch_dashboard" "main" {
     dashboard_name = "${var.project}-${var.environment}"
     
     dashboard_body = jsonencode({
       widgets = [
         {
           type = "metric"
           properties = {
             metrics = [
               ["AWS/ECS", "CPUUtilization", { stat = "Average" }],
               [".", "MemoryUtilization", { stat = "Average" }]
             ]
             period = 300
             stat   = "Average"
             region = var.aws_region
             title  = "ECS Resource Utilization"
           }
         },
         {
           type = "metric"
           properties = {
             metrics = [
               ["AWS/ApplicationELB", "TargetResponseTime"],
               [".", "RequestCount", { stat = "Sum" }],
               [".", "HTTPCode_Target_2XX_Count", { stat = "Sum" }],
               [".", "HTTPCode_Target_5XX_Count", { stat = "Sum" }]
             ]
             period = 300
             region = var.aws_region
             title  = "ALB Metrics"
           }
         }
       ]
     })
   }
   ```

2. **Alarms and Notifications**
   ```hcl
   # modules/monitoring/alarms.tf
   
   # SNS Topic for Alerts
   resource "aws_sns_topic" "alerts" {
     name = "${var.project}-${var.environment}-alerts"
     
     kms_master_key_id = aws_kms_key.main.id
   }
   
   resource "aws_sns_topic_subscription" "email" {
     topic_arn = aws_sns_topic.alerts.arn
     protocol  = "email"
     endpoint  = var.alert_email
   }
   
   # High CPU Alarm
   resource "aws_cloudwatch_metric_alarm" "high_cpu" {
     alarm_name          = "${var.project}-${var.environment}-high-cpu"
     comparison_operator = "GreaterThanThreshold"
     evaluation_periods  = "2"
     metric_name        = "CPUUtilization"
     namespace          = "AWS/ECS"
     period             = "300"
     statistic          = "Average"
     threshold          = "80"
     alarm_description  = "Triggers when CPU exceeds 80%"
     alarm_actions      = [aws_sns_topic.alerts.arn]
     
     dimensions = {
       ClusterName = aws_ecs_cluster.main.name
       ServiceName = aws_ecs_service.api_gateway.name
     }
   }
   
   # API Error Rate Alarm
   resource "aws_cloudwatch_metric_alarm" "api_errors" {
     alarm_name          = "${var.project}-${var.environment}-api-errors"
     comparison_operator = "GreaterThanThreshold"
     evaluation_periods  = "2"
     metric_name        = "HTTPCode_Target_5XX_Count"
     namespace          = "AWS/ApplicationELB"
     period             = "60"
     statistic          = "Sum"
     threshold          = "10"
     alarm_description  = "Triggers when 5XX errors exceed threshold"
     alarm_actions      = [aws_sns_topic.alerts.arn]
     
     dimensions = {
       LoadBalancer = aws_lb.main.arn_suffix
     }
   }
   
   # Availability Alarm
   resource "aws_cloudwatch_metric_alarm" "availability" {
     alarm_name          = "${var.project}-${var.environment}-availability"
     comparison_operator = "LessThanThreshold"
     evaluation_periods  = "2"
     metric_name        = "HealthyHostCount"
     namespace          = "AWS/ApplicationELB"
     period             = "60"
     statistic          = "Average"
     threshold          = "1"
     alarm_description  = "Triggers when no healthy targets"
     alarm_actions      = [aws_sns_topic.alerts.arn]
     
     dimensions = {
       TargetGroup  = aws_lb_target_group.api_gateway.arn_suffix
       LoadBalancer = aws_lb.main.arn_suffix
     }
   }
   ```

### Deliverables
- CloudWatch dashboards created
- Log groups configured
- Alarms and notifications set
- Monitoring coverage complete

### Testing Checklist
- [ ] Dashboards display metrics
- [ ] Logs being collected
- [ ] Alarms trigger correctly
- [ ] Notifications received

---

## Phase 7: CI/CD Integration
**Duration**: 2-3 days  
**Priority**: High

### Objectives
- Set up Terraform Cloud/GitHub Actions
- Implement deployment pipelines
- Configure environment promotion

### Tasks
1. **GitHub Actions Workflow**
   ```yaml
   # .github/workflows/terraform.yml
   name: Terraform Infrastructure
   
   on:
     push:
       branches:
         - main
         - develop
       paths:
         - 'infrastructure/**'
     pull_request:
       paths:
         - 'infrastructure/**'
   
   env:
     TF_VERSION: '1.5.0'
     TF_WORKSPACE: ${{ github.ref == 'refs/heads/main' && 'production' || 'staging' }}
   
   jobs:
     validate:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         
         - name: Setup Terraform
           uses: hashicorp/setup-terraform@v2
           with:
             terraform_version: ${{ env.TF_VERSION }}
         
         - name: Terraform Init
           run: terraform init
           working-directory: ./infrastructure/terraform
         
         - name: Terraform Format Check
           run: terraform fmt -check
           working-directory: ./infrastructure/terraform
         
         - name: Terraform Validate
           run: terraform validate
           working-directory: ./infrastructure/terraform
   
     plan:
       needs: validate
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         
         - name: Configure AWS Credentials
           uses: aws-actions/configure-aws-credentials@v2
           with:
             aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
             aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
             aws-region: us-east-1
         
         - name: Setup Terraform
           uses: hashicorp/setup-terraform@v2
           with:
             terraform_version: ${{ env.TF_VERSION }}
         
         - name: Terraform Init
           run: terraform init
           working-directory: ./infrastructure/terraform
         
         - name: Select Workspace
           run: terraform workspace select ${{ env.TF_WORKSPACE }} || terraform workspace new ${{ env.TF_WORKSPACE }}
           working-directory: ./infrastructure/terraform
         
         - name: Terraform Plan
           run: terraform plan -out=tfplan
           working-directory: ./infrastructure/terraform
         
         - name: Upload Plan
           uses: actions/upload-artifact@v3
           with:
             name: tfplan
             path: ./infrastructure/terraform/tfplan
   
     apply:
       needs: plan
       runs-on: ubuntu-latest
       if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop'
       environment:
         name: ${{ github.ref == 'refs/heads/main' && 'production' || 'staging' }}
       steps:
         - uses: actions/checkout@v3
         
         - name: Configure AWS Credentials
           uses: aws-actions/configure-aws-credentials@v2
           with:
             aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
             aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
             aws-region: us-east-1
         
         - name: Setup Terraform
           uses: hashicorp/setup-terraform@v2
           with:
             terraform_version: ${{ env.TF_VERSION }}
         
         - name: Terraform Init
           run: terraform init
           working-directory: ./infrastructure/terraform
         
         - name: Download Plan
           uses: actions/download-artifact@v3
           with:
             name: tfplan
             path: ./infrastructure/terraform
         
         - name: Terraform Apply
           run: terraform apply -auto-approve tfplan
           working-directory: ./infrastructure/terraform
   ```

2. **Environment Management**
   ```hcl
   # terraform/environments/production/terraform.tfvars
   environment = "production"
   aws_region  = "us-east-1"
   
   # Network
   vpc_cidr           = "10.0.0.0/16"
   availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
   enable_nat_gateway = true
   
   # Compute
   api_gateway_count = 3
   ml_service_count  = 2
   min_capacity      = 2
   max_capacity      = 10
   
   # Storage
   log_retention_days = 90
   
   # Monitoring
   alert_email = "ops-team@company.com"
   ```

### Deliverables
- CI/CD pipeline configured
- Environment-specific configs
- Automated deployments
- Rollback procedures

### Testing Checklist
- [ ] Pipeline runs successfully
- [ ] Infrastructure deploys
- [ ] Environment isolation works
- [ ] Rollback procedures tested

---

## Phase 8: Documentation and Handover
**Duration**: 2-3 days  
**Priority**: Medium

### Objectives
- Create operational documentation
- Implement disaster recovery
- Knowledge transfer

### Tasks
1. **Documentation**
   ```markdown
   # Infrastructure Documentation
   
   ## Architecture Overview
   - VPC with public/private subnets across 3 AZs
   - ECS Fargate for containerized services
   - Application Load Balancer for traffic distribution
   - S3 for ML model storage
   - CloudWatch for monitoring
   
   ## Access Management
   - IAM roles follow least privilege principle
   - Secrets stored in AWS Secrets Manager
   - KMS encryption for sensitive data
   
   ## Deployment Process
   1. Push code to GitHub
   2. GitHub Actions triggers Terraform
   3. Infrastructure updates applied
   4. Services deployed via ECS
   
   ## Monitoring
   - CloudWatch dashboards: [URL]
   - Alert notifications via SNS
   - Log aggregation in CloudWatch Logs
   
   ## Disaster Recovery
   - RTO: 1 hour
   - RPO: 24 hours
   - Backup strategy: S3 versioning, database snapshots
   ```

2. **Runbooks**
   ```markdown
   # Operational Runbooks
   
   ## Service Restart
   ```bash
   aws ecs update-service \
     --cluster production-cluster \
     --service api-gateway \
     --force-new-deployment
   ```
   
   ## Scale Service
   ```bash
   aws ecs update-service \
     --cluster production-cluster \
     --service api-gateway \
     --desired-count 5
   ```
   
   ## View Logs
   ```bash
   aws logs tail /aws/ecs/api-gateway \
     --follow \
     --filter-pattern ERROR
   ```
   ```

### Deliverables
- Complete documentation
- Operational runbooks
- Disaster recovery plan
- Team training completed

### Testing Checklist
- [ ] Documentation complete
- [ ] Runbooks validated
- [ ] DR plan tested
- [ ] Team trained

---

## Success Metrics

### Infrastructure Performance
- Deployment time: < 15 minutes
- Infrastructure cost: Within budget
- Availability: > 99.9%
- Security compliance: 100%

### Operational Excellence
- Automated deployments: 100%
- Infrastructure as Code: 100%
- Monitoring coverage: 100%
- Disaster recovery tested: Yes

---

## Risk Mitigation

### Technical Risks
1. **Terraform State Corruption**
   - Mitigation: State versioning, regular backups

2. **Service Unavailability**
   - Mitigation: Multi-AZ deployment, auto-scaling

3. **Security Breach**
   - Mitigation: WAF, security groups, encryption

### Operational Risks
1. **Cost Overruns**
   - Mitigation: Budget alerts, resource tagging

2. **Knowledge Gap**
   - Mitigation: Documentation, training

---

## Timeline Summary

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Foundation | 2-3 days | AWS Account |
| Phase 2: Networking | 3-4 days | Phase 1 |
| Phase 3: Compute | 4-5 days | Phase 2 |
| Phase 4: Storage | 3-4 days | Phase 1 |
| Phase 5: Security | 3-4 days | Phase 2-4 |
| Phase 6: Monitoring | 3-4 days | Phase 3-4 |
| Phase 7: CI/CD | 2-3 days | Phase 1-6 |
| Phase 8: Documentation | 2-3 days | All phases |

**Total Duration**: 22-30 days (4-6 weeks)

---

## Next Steps
1. Set up AWS accounts and permissions
2. Initialize Terraform backend
3. Begin Phase 1 implementation
4. Coordinate with development teams
5. Plan production deployment