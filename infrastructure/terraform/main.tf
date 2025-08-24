# Configure AWS Provider
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = var.common_tags
  }
}



# Configure Vercel Provider (for frontend deployment)
provider "vercel" {
  # Vercel token will be set via environment variable
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "amazon_linux_2" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# SSH Key Pair
resource "aws_key_pair" "main" {
  key_name   = var.ssh_key_name
  public_key = file("~/.ssh/${var.ssh_key_name}.pub")
}

# Networking Module
module "networking" {
  source = "./modules/networking"

  project     = var.project
  environment = var.environment
  vpc_cidr    = "10.0.0.0/16"
  azs         = data.aws_availability_zones.available.names
}

# Compute Module
module "compute" {
  source = "./modules/compute"

  project           = var.project
  environment       = var.environment
  vpc_id            = module.networking.vpc_id
  subnet_ids        = module.networking.public_subnet_ids
  security_group_id = module.networking.security_group_id
  key_name          = aws_key_pair.main.key_name
  ami_id            = data.aws_ami.amazon_linux_2.id
}

# Storage Module
module "storage" {
  source = "./modules/storage"

  project     = var.project
  environment = var.environment
}

# Monitoring Module
module "monitoring" {
  source = "./modules/monitoring"

  project           = var.project
  environment       = var.environment
  instance_ids      = module.compute.instance_ids
  security_group_id = module.networking.security_group_id
}
