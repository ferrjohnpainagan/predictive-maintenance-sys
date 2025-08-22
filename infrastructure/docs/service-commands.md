# Service Commands Reference

## Predictive Maintenance System - API Gateway & ML Service

### 🌏 **Infrastructure Details**

- **Region**: `ap-southeast-1` (Singapore)
- **VPC**: `vpc-0e405018dd7914d44`
- **Environment**: `free-tier`

---

## 🔑 **SSH Access Commands**

### **API Gateway Instance**

```bash
# SSH to API Gateway
ssh -i ~/.ssh/predictive-maintenance-key ec2-user@18.136.204.216

# Instance Details
Instance ID: i-0067148be3ef09ef3
Public IP: 18.136.204.216
Private IP: 10.0.0.x
Service Port: 3000
Health Check Port: 8080
```

### **ML Service Instance**

```bash
# SSH to ML Service
ssh -i ~/.ssh/predictive-maintenance-key ec2-user@13.215.159.154

# Instance Details
Instance ID: i-093a718e22c275662
Public IP: 13.215.159.154
Private IP: 10.0.0.x
Service Port: 8000
gRPC Port: 50051
Health Check Port: 8080
```

---

## 🐳 **Docker Commands**

### **Service Management**

```bash
# Check Docker status
sudo systemctl status docker

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# View container logs
docker logs <container_name>

# Follow container logs in real-time
docker logs -f <container_name>
```

### **Docker Compose Operations**

```bash
# Navigate to service directory
cd /opt/api-gateway    # For API Gateway
cd /opt/ml-service     # For ML Service

# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart services
docker-compose restart

# View service status
docker-compose ps

# View service logs
docker-compose logs

# Follow service logs
docker-compose logs -f
```

---

## 🔍 **Health Check Commands**

### **Service Health Monitoring**

```bash
# Check API Gateway health
curl -f http://18.136.204.216:3000/health
curl -f http://localhost:3000/health  # From within instance

# Check ML Service health
curl -f http://13.215.159.154:8000/health
curl -f http://localhost:8000/health  # From within instance

# Check health check logs
tail -f /var/log/health-check.log

# View cron job for health checks
crontab -l
```

### **Port Availability Checks**

```bash
# Check if ports are listening
netstat -tlnp | grep :3000    # API Gateway
netstat -tlnp | grep :8000    # ML Service
netstat -tlnp | grep :50051   # gRPC
netstat -tlnp | grep :8080    # Health Check

# Alternative using ss command
ss -tlnp | grep :3000
ss -tlnp | grep :8000
```

---

## 📊 **Monitoring & Logs**

### **CloudWatch Integration**

```bash
# View CloudWatch logs
aws logs describe-log-groups --region ap-southeast-1

# Get log events
aws logs get-log-events \
  --log-group-name "/aws/ec2/predictive-maintenance-free-tier-api-gateway" \
  --region ap-southeast-1

# View CloudWatch dashboard
aws cloudwatch describe-dashboards --region ap-southeast-1
```

### **System Monitoring**

```bash
# Check CPU usage
top
htop  # If installed

# Check memory usage
free -h

# Check disk usage
df -h

# Check system load
uptime

# Check running processes
ps aux | grep docker
ps aux | grep node  # For API Gateway
ps aux | grep python # For ML Service
```

---

## 🚀 **Deployment Commands**

### **Manual Service Deployment**

```bash
# Stop existing containers
docker-compose down

# Pull latest images
docker-compose pull

# Start services with new images
docker-compose up -d

# Verify deployment
docker-compose ps
curl -f http://localhost:3000/health  # API Gateway
curl -f http://localhost:8000/health  # ML Service
```

### **Rollback Commands**

```bash
# List available images
docker images

# Rollback to previous image
docker-compose down
docker tag <previous_image> <current_tag>
docker-compose up -d

# Check rollback success
docker-compose ps
curl -f http://localhost:3000/health
```

---

## 🛠️ **Troubleshooting Commands**

### **Service Debugging**

```bash
# Check service status
sudo systemctl status docker

# Restart Docker service
sudo systemctl restart docker

# Check Docker daemon logs
sudo journalctl -u docker

# Inspect container configuration
docker inspect <container_name>

# Execute commands in running container
docker exec -it <container_name> /bin/bash
```

### **Network Troubleshooting**

```bash
# Test internal connectivity
ping 10.0.0.x  # Between instances

# Check security group rules
aws ec2 describe-security-groups \
  --group-ids sg-0042dbc65f2fe6753 \
  --region ap-southeast-1

# Test external connectivity
curl -I https://www.google.com
nslookup google.com
```

---

## 📁 **File System Commands**

### **Service Directories**

```bash
# API Gateway files
ls -la /opt/api-gateway/
cat /opt/api-gateway/docker-compose.yml

# ML Service files
ls -la /opt/ml-service/
cat /opt/ml-service/docker-compose.yml

# User data script logs
tail -f /var/log/startup.log
tail -f /var/log/health-check.log
```

### **Configuration Management**

```bash
# View environment variables
env | grep -i api
env | grep -i ml

# Check service configuration
cat /opt/api-gateway/.env  # If exists
cat /opt/ml-service/.env   # If exists
```

---

## 🔧 **Maintenance Commands**

### **System Updates**

```bash
# Update system packages
sudo yum update -y

# Update Docker
sudo yum update docker

# Clean up Docker resources
docker system prune -f
docker volume prune -f
```

### **Backup Operations**

```bash
# Backup service configurations
cp -r /opt/api-gateway /opt/backup/api-gateway-$(date +%Y%m%d)
cp -r /opt/ml-service /opt/backup/ml-service-$(date +%Y%m%d)

# Backup logs
cp /var/log/startup.log /opt/backup/startup-$(date +%Y%m%d).log
cp /var/log/health-check.log /opt/backup/health-check-$(date +%Y%m%d).log
```

---

## 📋 **Quick Reference Cheat Sheet**

### **Daily Operations**

```bash
# Check all services status
docker ps && docker-compose ps

# View recent logs
docker-compose logs --tail=50

# Health check all services
curl -f http://18.136.204.216:3000/health && \
curl -f http://13.215.159.154:8000/health

# SSH to both instances
ssh -i ~/.ssh/predictive-maintenance-key ec2-user@18.136.204.216
ssh -i ~/.ssh/predictive-maintenance-key ec2-user@13.215.159.154
```

### **Emergency Commands**

```bash
# Restart all services
docker-compose restart

# Stop all services
docker-compose down

# Start all services
docker-compose up -d

# Check system resources
top && df -h && free -h
```

---

## 🚨 **Important Notes**

1. **SSH Key Location**: `~/.ssh/predictive-maintenance-key`
2. **Service User**: `ec2-user`
3. **Docker Requires Sudo**: Use `sudo` for Docker commands if needed
4. **Health Check Frequency**: Every 5 minutes via cron
5. **Log Retention**: 7 days in CloudWatch
6. **Backup Strategy**: Manual backups recommended before major changes

---

## 🔗 **Useful URLs**

- **API Gateway**: http://18.136.204.216:3000
- **ML Service**: http://13.215.159.154:8000
- **Health Checks**: Port 8080 on both instances
- **gRPC**: Port 50051 on ML Service instance

---

_Last Updated: $(date)_
_Infrastructure Version: Free Tier_
_Region: ap-southeast-1 (Singapore)_
