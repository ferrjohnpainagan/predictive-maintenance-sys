#!/bin/bash

# Update system
yum update -y

# Install Python 3.9 and pip
yum install -y python3 python3-pip

# Install Docker
yum install -y docker
systemctl start docker
systemctl enable docker

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create service directory
mkdir -p /opt/ml-service
cd /opt/ml-service

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  ml-service:
    image: your-registry/ml-service:latest
    ports:
      - "8000:8000"
      - "50051:50051"
    environment:
      - ENVIRONMENT=${environment}
      - PORT=8000
      - GRPC_PORT=50051
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
EOF

# Create health check script
cat > /opt/health-check.sh << 'EOF'
#!/bin/bash

# Check if service is running
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
  echo "$(date): ML Service is healthy" >> /var/log/health-check.log
  exit 0
else
  echo "$(date): ML Service is unhealthy" >> /var/log/health-check.log
  exit 1
fi
EOF

chmod +x /opt/health-check.sh

# Add health check to crontab
echo "*/5 * * * * /opt/health-check.sh" | crontab -

# Start service
docker-compose up -d

# Log startup
echo "$(date): ML Service started" >> /var/log/startup.log
