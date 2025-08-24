# API Gateway EC2 Instance
resource "aws_instance" "api_gateway" {
  instance_type = "t3.micro"  # Free tier eligible in ap-southeast-1
  ami           = var.ami_id

  vpc_security_group_ids = [var.security_group_id]
  subnet_id              = var.subnet_ids[0]

  key_name = var.key_name

  user_data = templatefile("${path.module}/user_data/api_gateway.sh", {
    service_name = "api-gateway"
    environment  = var.environment
  })

  tags = {
    Name = "${var.project}-${var.environment}-api-gateway"
    Role = "api-gateway"
  }
}

# ML Service EC2 Instance
resource "aws_instance" "ml_service" {
  instance_type = "t3.micro"  # Free tier eligible in ap-southeast-1
  ami           = var.ami_id

  vpc_security_group_ids = [var.security_group_id]
  subnet_id              = var.subnet_ids[0]

  key_name = var.key_name

  user_data = templatefile("${path.module}/user_data/ml_service.sh", {
    service_name = "ml-service"
    environment  = var.environment
  })

  tags = {
    Name = "${var.project}-${var.environment}-ml-service"
    Role = "ml-service"
  }
}
