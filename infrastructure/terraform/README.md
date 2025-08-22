# Predictive Maintenance System - Infrastructure as Code

This directory contains the Terraform configuration for deploying the predictive maintenance system on AWS Free Tier.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │   ML Service    │    │   S3 Storage    │
│   (EC2 t2.micro)│    │   (EC2 t2.micro)│    │   (ML Data)     │
│   Port: 3000    │    │   Port: 8000    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   VPC Network   │
                    │   (10.0.0.0/16)│
                    └─────────────────┘
```

## 🚀 Quick Start

### Current Infrastructure Status

**✅ Infrastructure Deployed Successfully!**

| **Component**   | **Status**    | **Details**                                |
| --------------- | ------------- | ------------------------------------------ |
| **VPC**         | ✅ Running    | `vpc-0e405018dd7914d44` (Singapore)        |
| **API Gateway** | ✅ Running    | `18.136.204.216:3000`                      |
| **ML Service**  | ✅ Running    | `13.215.159.154:8000`                      |
| **S3 Storage**  | ✅ Created    | `predictive-maintenance-free-tier-ml-data` |
| **Monitoring**  | ✅ Configured | CloudWatch dashboard & alarms              |

**Connection Commands:**

```bash
# API Gateway
ssh -i ~/.ssh/predictive-maintenance-key ec2-user@18.136.204.216

# ML Service
ssh -i ~/.ssh/predictive-maintenance-key ec2-user@13.215.159.154
```

### Prerequisites

1. **AWS CLI configured** with your free tier account
2. **Terraform installed** (version >= 1.5.0)
3. **SSH key pair** generated
4. **Supabase project** created

### Setup Steps

1. **Generate SSH Key Pair**

   ```bash
   # Generate SSH key pair in ~/.ssh/
   ssh-keygen -t rsa -b 4096 -f ~/.ssh/predictive-maintenance-key -N ""
   chmod 600 ~/.ssh/predictive-maintenance-key
   chmod 644 ~/.ssh/predictive-maintenance-key.pub
   ```

2. **Configure Variables**

   ```bash
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your values
   ```

3. **Initialize Terraform**

   ```bash
   terraform init
   ```

4. **Plan Deployment**

   ```bash
   terraform plan
   ```

5. **Deploy Infrastructure**
   ```bash
   terraform apply
   ```

## 📁 Project Structure

```
terraform/
├── main.tf                 # Main configuration
├── variables.tf            # Variable definitions
├── outputs.tf              # Output values
├── versions.tf             # Provider versions
├── terraform.tfvars.example # Example variables
└── modules/                # Terraform modules
    ├── networking/         # VPC, subnets, security groups
    ├── compute/            # EC2 instances
    ├── storage/            # S3 buckets
    └── monitoring/         # CloudWatch configuration
```

## 📚 Documentation

### Service Commands

- **[Service Commands Reference](../docs/service-commands.md)** - Comprehensive command reference for API Gateway and ML Service
- **[Quick Commands Card](../docs/quick-commands.md)** - Essential commands for daily operations

### Infrastructure Management

- **Terraform State**: Local state backend for development
- **Module Documentation**: Each module includes its own README
- **User Data Scripts**: Automated instance setup scripts

## 🔧 Modules

### Networking Module

- **VPC**: 10.0.0.0/16 with single public subnet
- **Security Groups**: SSH (22), API Gateway (3000), ML Service (8000), gRPC (50051)
- **Internet Gateway**: For public internet access

### Compute Module

- **API Gateway**: EC2 t2.micro instance on port 3000
- **ML Service**: EC2 t2.micro instance on ports 8000 and 50051
- **User Data Scripts**: Automatic Docker installation and service setup

### Storage Module

- **S3 Bucket**: For ML model artifacts, training data, and logs
- **Lifecycle Policies**: Cost optimization for free tier
- **Encryption**: Server-side encryption enabled

### Monitoring Module

- **CloudWatch Log Groups**: For service logs
- **Metric Alarms**: CPU utilization monitoring
- **Dashboard**: Basic monitoring dashboard

## 💰 Free Tier Considerations

- **EC2**: 750 hours/month of t2.micro (1 vCPU, 1GB RAM)
- **S3**: 5GB storage, 20K GET requests, 2K PUT requests
- **CloudWatch**: Basic metrics, 5GB logs
- **Data Transfer**: 15GB outbound free

## 🔐 Security Features

- **SSH Keys**: Stored securely in ~/.ssh/ (not in repository)
- **Security Groups**: Minimal required ports open
- **S3 Encryption**: Server-side encryption enabled
- **IAM Roles**: Minimal required permissions

## 📊 Outputs

After successful deployment, Terraform will output:

- **VPC and Subnet IDs**: Network configuration
- **EC2 Public IPs**: Service access endpoints
- **S3 Bucket Name**: ML data storage location
- **Connection Information**: SSH commands and service URLs

## 🚨 Monitoring & Alerts

- **CPU Alarms**: Alert when CPU > 80% for 10 minutes
- **Health Checks**: Automated service health monitoring
- **Log Retention**: 7 days (free tier optimized)

## 🧹 Cleanup

To destroy all resources and avoid charges:

```bash
terraform destroy
```

**⚠️ Warning**: This will delete all created resources!

## 🔄 Next Steps

After infrastructure deployment:

1. **Deploy Services**: Build and deploy API Gateway and ML Service
2. **Configure CI/CD**: Set up GitHub Actions for automated deployment
3. **Set Up Monitoring**: Configure additional CloudWatch metrics
4. **Test Integration**: Verify services can communicate

## 📚 Additional Resources

- [AWS Free Tier](https://aws.amazon.com/free/)
- [Terraform Documentation](https://www.terraform.io/docs)
- [EC2 User Data](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html)
- [CloudWatch Monitoring](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/)

## 🆘 Troubleshooting

### Common Issues

1. **SSH Connection Failed**

   - Verify security group allows port 22
   - Check SSH key permissions (600)
   - Ensure instance is running

2. **Service Not Accessible**

   - Verify security group allows required ports
   - Check user data script execution
   - Review CloudWatch logs

3. **Terraform State Issues**
   - Use `terraform refresh` to sync state
   - Check AWS console for resource status
   - Verify IAM permissions

### Support

For issues related to:

- **Infrastructure**: Check Terraform logs and AWS console
- **Services**: Review EC2 user data and Docker logs
- **Networking**: Verify VPC and security group configuration
