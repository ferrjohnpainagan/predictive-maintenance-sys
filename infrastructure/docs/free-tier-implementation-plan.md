# Free Tier Infrastructure Implementation Plan - Predictive Maintenance System

## Project Overview

Implementation plan for building the complete predictive maintenance system infrastructure using AWS Free Tier services. This approach uses EC2 instances instead of ECS Fargate to stay within free tier limits while maintaining full functionality.

## Technology Stack (Free Tier)

- **IaC Tools**: Terraform
- **Cloud Providers**: AWS (Free Tier), Supabase, Vercel
- **Compute**: EC2 t2.micro instances (Free Tier eligible)
- **Storage**: S3 (5GB free), RDS t2.micro (Free Tier eligible)
- **CI/CD**: GitHub Actions
- **Secret Management**: GitHub Secrets, environment files
- **Monitoring**: Basic CloudWatch (Free Tier)

## Free Tier Limits & Constraints

- **EC2**: 750 hours/month of t2.micro (1 vCPU, 1GB RAM)
- **S3**: 5GB storage, 20K GET requests, 2K PUT requests
- **RDS**: 750 hours/month of db.t2.micro
- **CloudWatch**: Basic metrics, 5GB logs
- **Data Transfer**: 15GB outbound free

---

## Phase 1: Foundation and Prerequisites

**Duration**: 2-3 days  
**Priority**: Critical

### Objectives

- Set up AWS Free Tier account
- Configure Terraform for free tier deployment
- Establish basic project structure

### Tasks

1. **AWS Free Tier Setup**

   ```bash
   # Create AWS account with free tier eligibility
   # Enable billing alerts (critical for free tier)
   # Set up IAM user with minimal permissions
   ```

2. **Terraform Configuration**

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

     backend "local" {
       # Use local state for free tier (no S3 costs)
       path = "terraform.tfstate"
     }
   }
   ```

3. **Project Structure**
   ```
   infrastructure/
   ├── terraform/
   │   ├── free-tier/
   │   │   ├── main.tf
   │   │   ├── variables.tf
   │   │   ├── outputs.tf
   │   │   └── user_data/
   │   │       ├── api_gateway.sh
   │   │       └── ml_service.sh
   │   ├── modules/
   │   │   ├── networking/
   │   │   ├── compute/
   │   │   ├── database/
   │   │   └── storage/
   │   └── scripts/
   └── docs/
   ```

### Deliverables

- AWS Free Tier account configured
- Terraform initialized with local state
- Project structure established

---

## Phase 2: Basic Networking (Free Tier)

**Duration**: 2-3 days  
**Priority**: High

### Objectives

- Create minimal VPC for free tier deployment
- Set up security groups
- Configure basic routing

### Tasks

1. **VPC Configuration**

   ```hcl
   # modules/networking/vpc.tf
   resource "aws_vpc" "main" {
     cidr_block = "10.0.0.0/16"

     tags = {
       Name = "${var.project}-${var.environment}-vpc"
       Environment = var.environment
     }
   }

   # Single public subnet (free tier approach)
   resource "aws_subnet" "public" {
     vpc_id            = aws_vpc.main.id
     cidr_block        = "10.0.1.0/24"
     availability_zone = data.aws_availability_zones.available.names[0]

     tags = {
       Name = "${var.project}-${var.environment}-public"
     }
   }
   ```

2. **Security Groups**

   ```hcl
   # modules/networking/security_groups.tf
   resource "aws_security_group" "ec2" {
     name_prefix = "${var.project}-${var.environment}-ec2-"
     vpc_id      = aws_vpc.main.id

     ingress {
       from_port   = 22
       to_port     = 22
       protocol    = "tcp"
       cidr_blocks = ["0.0.0.0/0"]  # SSH access
     }

     ingress {
       from_port   = 3000
       to_port     = 3000
       protocol    = "tcp"
       cidr_blocks = ["0.0.0.0/0"]  # API Gateway
     }

     ingress {
       from_port   = 50051
       to_port     = 50051
       protocol    = "tcp"
       cidr_blocks = ["10.0.1.0/24"]  # gRPC internal
     }

     egress {
       from_port   = 0
       to_port     = 0
       protocol    = "-1"
       cidr_blocks = ["0.0.0.0/0"]
     }
   }
   ```

3. **Internet Gateway**

   ```hcl
   resource "aws_internet_gateway" "main" {
     vpc_id = aws_vpc.main.id

     tags = {
       Name = "${var.project}-${var.environment}-igw"
     }
   }

   resource "aws_route_table" "public" {
     vpc_id = aws_vpc.main.id

     route {
       cidr_block = "0.0.0.0/0"
       gateway_id = aws_internet_gateway.main.id
     }
   }
   ```

### Deliverables

- Basic VPC with single public subnet
- Security groups configured
- Internet connectivity established

---

## Phase 3: Compute Infrastructure (Free Tier)

**Duration**: 3-4 days  
**Priority**: High

### Objectives

- Deploy EC2 instances for services
- Configure Docker and services
- Set up basic monitoring

### Tasks

1. **EC2 Instance for API Gateway**

   ```hcl
   # modules/compute/ec2.tf
   resource "aws_instance" "api_gateway" {
     instance_type = "t2.micro"  # Free tier eligible
     ami           = data.aws_ami.amazon_linux_2.id

     vpc_security_group_ids = [aws_security_group.ec2.id]
     subnet_id              = aws_subnet.public[0].id

     key_name = aws_key_pair.main.key_name

     user_data = templatefile("${path.module}/user_data/api_gateway.sh", {
       service_name = "api-gateway"
       docker_image = "your-registry/api-gateway:latest"
       environment = var.environment
     })

     tags = {
       Name = "${var.project}-${var.environment}-api-gateway"
     }
   }
   ```

2. **EC2 Instance for ML Service**

   ```hcl
   resource "aws_instance" "ml_service" {
     instance_type = "t2.micro"  # Free tier eligible
     ami           = data.aws_ami.amazon_linux_2.id

     vpc_security_group_ids = [aws_security_group.ec2.id]
     subnet_id              = aws_subnet.public[0].id

     key_name = aws_key_pair.main.key_name

     user_data = templatefile("${path.module}/user_data/ml_service.sh", {
       service_name = "ml-service"
       docker_image = "your-registry/ml-service:latest"
       environment = var.environment
     })

     tags = {
       Name = "${var.project}-${var.environment}-ml-service"
     }
   }
   ```

3. **User Data Scripts**

   ```bash
   # user_data/api_gateway.sh
   #!/bin/bash
   yum update -y
   yum install -y docker
   systemctl start docker
   systemctl enable docker

   # Install Docker Compose
   curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   chmod +x /usr/local/bin/docker-compose

   # Create service directory
   mkdir -p /opt/api-gateway
   cd /opt/api-gateway

   # Create docker-compose.yml
   cat > docker-compose.yml << 'EOF'
   version: '3.8'
   services:
     api-gateway:
       image: ${docker_image}
       ports:
         - "3000:3000"
       environment:
         - NODE_ENV=${environment}
       restart: unless-stopped
   EOF

   # Start service
   docker-compose up -d
   ```

### Deliverables

- EC2 instances deployed and running
- Services accessible via public IPs
- Basic monitoring configured

---

## Phase 4: Storage & Database (Free Tier)

**Duration**: 2-3 days  
**Priority**: Medium

### Objectives

- Set up S3 for data storage
- Configure RDS database
- Implement backup strategy

### Tasks

1. **S3 Configuration**

   ```hcl
   # modules/storage/s3.tf
   resource "aws_s3_bucket" "data" {
     bucket = "${var.project}-${var.environment}-data"

     tags = {
       Name = "${var.project}-${var.environment}-data"
     }
   }

   resource "aws_s3_bucket_versioning" "data" {
     bucket = aws_s3_bucket.data.id
     versioning_configuration {
       status = "Enabled"
     }
   }

   resource "aws_s3_bucket_lifecycle_configuration" "data" {
     bucket = aws_s3_bucket.data.id

     rule {
       id     = "free_tier_optimization"
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
   ```

2. **RDS Database**

   ```hcl
   # modules/database/rds.tf
   resource "aws_db_instance" "main" {
     identifier = "${var.project}-${var.environment}-db"

     engine         = "postgres"
     engine_version = "13.7"
     instance_class = "db.t2.micro"  # Free tier eligible

     allocated_storage     = 20
     max_allocated_storage = 100
     storage_encrypted     = false  # Free tier limitation

     db_name  = "predictive_maintenance"
     username = var.db_username
     password = var.db_password

     vpc_security_group_ids = [aws_security_group.rds.id]
     db_subnet_group_name   = aws_db_subnet_group.main.name

     backup_retention_period = 7
     backup_window          = "03:00-04:00"
     maintenance_window     = "sun:04:00-sun:05:00"

     tags = {
       Name = "${var.project}-${var.environment}-database"
     }
   }
   ```

### Deliverables

- S3 bucket configured with lifecycle policies
- RDS database running
- Backup procedures documented

---

## Phase 5: CI/CD for Free Tier

**Duration**: 2-3 days  
**Priority**: Medium

### Objectives

- Set up GitHub Actions for EC2 deployment
- Configure automated testing
- Implement deployment procedures

### Tasks

1. **GitHub Actions Workflow**

   ```yaml
   # .github/workflows/free-tier-deploy.yml
   name: Free Tier Deployment

   on:
     push:
       branches: [main, develop]
       paths:
         - "backend/**"
         - "infrastructure/**"

   jobs:
     deploy:
       runs-on: ubuntu-latest

       steps:
         - uses: actions/checkout@v3

         - name: Configure AWS Credentials
           uses: aws-actions/configure-aws-credentials@v2
           with:
             aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
             aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
             aws-region: us-east-1

         - name: Build and Push Docker Images
           run: |
             # Build images
             docker build -t your-registry/api-gateway:${{ github.sha }} ./backend/api-gateway
             docker build -t your-registry/ml-service:${{ github.sha }} ./backend/ml-service

             # Push to registry
             docker push your-registry/api-gateway:${{ github.sha }}
             docker push your-registry/ml-service:${{ github.sha }}

         - name: Deploy to EC2
           run: |
             # Deploy API Gateway
             ssh -i ${{ secrets.EC2_SSH_KEY }} ec2-user@${{ secrets.EC2_API_HOST }} \
               "cd /opt/api-gateway && \
                docker-compose pull && \
                docker-compose up -d"

             # Deploy ML Service
             ssh -i ${{ secrets.EC2_SSH_KEY }} ec2-user@${{ secrets.EC2_ML_HOST }} \
               "cd /opt/ml-service && \
                docker-compose pull && \
                docker-compose up -d"
   ```

2. **Deployment Scripts**

   ```bash
   # scripts/deploy-free-tier.sh
   #!/bin/bash

   SERVICE=$1
   VERSION=$2

   case $SERVICE in
     "api-gateway")
       HOST=$EC2_API_HOST
       SERVICE_DIR="/opt/api-gateway"
       ;;
     "ml-service")
       HOST=$EC2_ML_HOST
       SERVICE_DIR="/opt/ml-service"
       ;;
     *)
       echo "Unknown service: $SERVICE"
       exit 1
       ;;
   esac

   ssh -i $EC2_SSH_KEY ec2-user@$HOST \
     "cd $SERVICE_DIR && \
      docker-compose pull && \
      docker-compose up -d"
   ```

### Deliverables

- Automated deployment pipeline
- Deployment scripts and documentation
- Testing procedures

---

## Phase 6: Monitoring & Observability (Free Tier)

**Duration**: 2-3 days  
**Priority**: Low

### Objectives

- Set up basic CloudWatch monitoring
- Configure logging
- Implement health checks

### Tasks

1. **CloudWatch Monitoring**

   ```hcl
   # modules/monitoring/cloudwatch.tf
   resource "aws_cloudwatch_log_group" "api_gateway" {
     name              = "/aws/ec2/${var.project}-api-gateway"
     retention_in_days = 7  # Free tier optimization
   }

   resource "aws_cloudwatch_log_group" "ml_service" {
     name              = "/aws/ec2/${var.project}-ml-service"
     retention_in_days = 7
   }

   resource "aws_cloudwatch_metric_alarm" "cpu_high" {
     alarm_name          = "${var.project}-${var.environment}-cpu-high"
     comparison_operator = "GreaterThanThreshold"
     evaluation_periods  = 2
     metric_name         = "CPUUtilization"
     namespace           = "AWS/EC2"
     period              = 300
     statistic           = "Average"
     threshold           = 80

     dimensions = {
       InstanceId = aws_instance.api_gateway.id
     }
   }
   ```

2. **Health Check Endpoints**

   ```bash
   # Add to user data scripts
   # Create health check endpoint
   cat > /opt/health-check.sh << 'EOF'
   #!/bin/bash

   # Check if service is running
   if curl -f http://localhost:3000/health > /dev/null 2>&1; then
     echo "Service is healthy"
     exit 0
   else
     echo "Service is unhealthy"
     exit 1
   fi
   EOF

   chmod +x /opt/health-check.sh

   # Add to crontab
   echo "*/5 * * * * /opt/health-check.sh" | crontab -
   ```

### Deliverables

- Basic monitoring configured
- Health checks implemented
- Logging setup

---

## Success Metrics (Free Tier)

### Infrastructure Performance

- **Cost**: $0/month (within free tier limits)
- **Availability**: > 95% (single instance)
- **Response Time**: < 500ms (basic monitoring)

### Operational Excellence

- **Deployment Time**: < 10 minutes
- **Manual Operations**: Acceptable for free tier
- **Monitoring Coverage**: Basic (CPU, memory, logs)

---

## Migration Preparation Checklist

### Month 6-8: Performance Evaluation

- [ ] Monitor resource usage
- [ ] Test with realistic data volumes
- [ ] Identify bottlenecks
- [ ] Document scaling requirements

### Month 9-10: Migration Preparation

- [ ] Containerize all services
- [ ] Implement health checks
- [ ] Test backup/restore procedures
- [ ] Document deployment procedures

### Month 11: Migration Planning

- [ ] Design production architecture
- [ ] Plan migration timeline
- [ ] Prepare rollback procedures
- [ ] Train team on new procedures

---

## Timeline Summary

| Phase               | Duration | Dependencies          |
| ------------------- | -------- | --------------------- |
| Phase 1: Foundation | 2-3 days | AWS Free Tier Account |
| Phase 2: Networking | 2-3 days | Phase 1               |
| Phase 3: Compute    | 3-4 days | Phase 2               |
| Phase 4: Storage    | 2-3 days | Phase 1               |
| Phase 5: CI/CD      | 2-3 days | Phase 3               |
| Phase 6: Monitoring | 2-3 days | Phase 3               |

**Total Duration**: 13-19 days (2-3 weeks)

---

## Next Steps

1. **Immediate**: Set up AWS Free Tier account
2. **Week 1**: Initialize Terraform and basic networking
3. **Week 2**: Deploy EC2 instances and services
4. **Week 3**: Set up CI/CD and monitoring
5. **Month 2-6**: Test and optimize system
6. **Month 6-10**: Prepare for migration
7. **Month 12**: Execute migration to paid services

This free tier approach allows you to build and test your complete system while staying within AWS free tier limits, with a clear path to scale to production-grade infrastructure when ready.
