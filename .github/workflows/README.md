# 🚀 GitHub Actions CI/CD Setup Guide

## Predictive Maintenance System - Automated Deployment

This guide explains how to set up and configure the CI/CD pipeline for automatically deploying your predictive maintenance system to AWS EC2 instances.

---

## 📋 **Prerequisites**

### **1. GitHub Repository Setup**

- ✅ Repository with `main` branch
- ✅ Code in `backend/`, `mlops/`, and `shared/` directories
- ✅ GitHub Actions enabled

### **2. Infrastructure Requirements**

- ✅ AWS EC2 instances running (API Gateway & ML Service)
- ✅ SSH access configured
- ✅ Docker installed on EC2 instances
- ✅ Health check endpoints working

### **3. Docker Registry**

- ✅ Docker Hub account, or
- ✅ AWS ECR repository, or
- ✅ Other container registry

---

## 🔐 **GitHub Secrets Configuration**

### **Required Secrets**

Navigate to your GitHub repository → **Settings** → **Secrets and variables** → **Actions**

#### **EC2 Instance Details**

```bash
# Secret Name: EC2_API_GATEWAY_IP
# Secret Value: 18.136.204.216

# Secret Name: EC2_ML_SERVICE_IP
# Secret Value: 13.215.159.154
```

#### **SSH Authentication**

```bash
# Secret Name: SSH_PRIVATE_KEY
# Secret Value: [Your private SSH key content]
# Note: Include the entire private key including BEGIN/END lines
```

#### **Docker Registry**

```bash
# Secret Name: DOCKER_REGISTRY
# Secret Value: [Your registry URL]
# Examples:
# - docker.io/yourusername
# - your-account.dkr.ecr.ap-southeast-1.amazonaws.com
# - ghcr.io/yourusername

# Secret Name: DOCKER_USERNAME
# Secret Value: [Your registry username]

# Secret Name: DOCKER_PASSWORD
# Secret Value: [Your registry password/token]
```

---

## 🐳 **Docker Registry Setup**

### **Option 1: Docker Hub (Recommended for Free Tier)**

1. **Create Docker Hub Account**

   ```bash
   # Visit: https://hub.docker.com/
   # Create account and verify email
   ```

2. **Create Repository**

   ```bash
   # Create repositories:
   # - yourusername/api-gateway
   # - yourusername/ml-service
   ```

3. **Configure GitHub Secrets**
   ```bash
   DOCKER_REGISTRY: docker.io/yourusername
   DOCKER_USERNAME: yourusername
   DOCKER_PASSWORD: your-access-token
   ```

### **Option 2: AWS ECR (Production Ready)**

1. **Create ECR Repositories**

   ```bash
   aws ecr create-repository --repository-name api-gateway --region ap-southeast-1
   aws ecr create-repository --repository-name ml-service --region ap-southeast-1
   ```

2. **Get ECR Login Token**

   ```bash
   aws ecr get-login-password --region ap-southeast-1
   ```

3. **Configure GitHub Secrets**
   ```bash
   DOCKER_REGISTRY: your-account.dkr.ecr.ap-southeast-1.amazonaws.com
   DOCKER_USERNAME: AWS
   DOCKER_PASSWORD: [ECR login token]
   ```

### **Option 3: GitHub Container Registry (GHCR)**

1. **Enable GHCR**

   ```bash
   # Go to repository Settings → Packages
   # Enable GitHub Packages
   ```

2. **Configure GitHub Secrets**
   ```bash
   DOCKER_REGISTRY: ghcr.io/yourusername
   DOCKER_USERNAME: yourusername
   DOCKER_PASSWORD: [GitHub Personal Access Token]
   ```

---

## 🔧 **Workflow Configuration**

### **Automatic Triggers**

The workflow automatically runs when:

- ✅ **Push to main/develop** branches
- ✅ **Changes in backend/, mlops/, shared/** directories
- ✅ **Pull requests** to main branch

### **Manual Triggers**

You can manually trigger deployments:

1. Go to **Actions** tab
2. Select **Deploy Predictive Maintenance System**
3. Click **Run workflow**
4. Choose:
   - **Environment**: free-tier, staging, production
   - **Services**: all, api-gateway, ml-service

---

## 🚀 **Deployment Process**

### **Phase 1: Build & Test**

```yaml
1. Checkout code
2. Setup Node.js (API Gateway)
3. Setup Python (ML Service)
4. Install dependencies
5. Run tests
6. Security scans
```

### **Phase 2: Docker Build**

```yaml
1. Build API Gateway image
2. Build ML Service image
3. Push to registry
4. Generate deployment manifest
```

### **Phase 3: Deployment**

```yaml
1. Deploy to API Gateway EC2
2. Deploy to ML Service EC2
3. Health checks
4. Verification
```

### **Phase 4: Post-Deployment**

```yaml
1. Verify both services
2. Generate deployment report
3. Update GitHub summary
```

---

## 📊 **Monitoring & Health Checks**

### **Health Check Endpoints**

```bash
# API Gateway
curl -f http://18.136.204.216:3000/health

# ML Service
curl -f http://13.215.159.154:8000/health
```

### **Deployment Logs**

```bash
# On EC2 instances
tail -f /var/log/deployment.log
tail -f /var/log/health-check.log
```

### **GitHub Actions Logs**

- View real-time logs in **Actions** tab
- Download logs for debugging
- Check deployment status

---

## 🛠️ **Troubleshooting**

### **Common Issues & Solutions**

#### **1. SSH Connection Failed**

```bash
# Check SSH key format
# Ensure private key includes BEGIN/END lines
# Verify EC2 IP addresses are correct
# Check security group allows SSH (port 22)
```

#### **2. Docker Build Failed**

```bash
# Verify Docker registry credentials
# Check Dockerfile syntax
# Ensure build context is correct
# Check registry permissions
```

#### **3. Health Check Failed**

```bash
# Verify health endpoint exists in your app
# Check if port is accessible
# Verify Docker container is running
# Check container logs
```

#### **4. Deployment Timeout**

```bash
# Increase health check timeout in workflow
# Check EC2 instance resources
# Verify network connectivity
# Check Docker daemon status
```

### **Debug Commands**

```bash
# On EC2 instances
docker ps                    # Check running containers
docker logs <container>      # View container logs
docker-compose ps           # Check service status
curl -f localhost:3000/health  # Test health locally
```

---

## 🔄 **Rollback Procedures**

### **Automatic Rollback**

The workflow includes automatic rollback capabilities:

1. **Backup creation** before each deployment
2. **Health check verification** after deployment
3. **Manual rollback trigger** via workflow dispatch

### **Manual Rollback**

```bash
# SSH to EC2 instance
cd /opt/api-gateway  # or /opt/ml-service

# List backup files
ls -la docker-compose.yml.backup.*

# Restore from backup
cp docker-compose.yml.backup.20241222_143022 docker-compose.yml

# Restart service
docker-compose down
docker-compose up -d
```

---

## 📈 **Performance Optimization**

### **Build Caching**

- ✅ **GitHub Actions Cache**: npm, pip dependencies
- ✅ **Docker Layer Caching**: Buildx with GHA cache
- ✅ **Parallel Jobs**: Independent service deployments

### **Deployment Speed**

- ✅ **Health Check Optimization**: 30-second intervals
- ✅ **Parallel Verification**: Both services verified simultaneously
- ✅ **Efficient SSH**: Single connection per deployment

---

## 🔒 **Security Considerations**

### **Secrets Management**

- ✅ **GitHub Secrets**: Encrypted at rest
- ✅ **SSH Keys**: Private keys never logged
- ✅ **Registry Credentials**: Secure authentication

### **Network Security**

- ✅ **SSH Only**: No HTTP deployment endpoints
- ✅ **Health Checks**: Internal verification only
- ✅ **Backup Security**: Local EC2 storage

---

## 📚 **Additional Resources**

### **Documentation**

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [SSH Action Documentation](https://github.com/appleboy/ssh-action)

### **Support**

- Check GitHub Actions logs for detailed error messages
- Verify all secrets are correctly configured
- Test SSH connectivity manually before deployment
- Review EC2 instance logs for application-level issues

---

## 🎯 **Next Steps**

1. ✅ **Configure GitHub Secrets** (Required)
2. ✅ **Set up Docker Registry** (Required)
3. ✅ **Test workflow** with small changes
4. ✅ **Monitor deployments** and logs
5. ✅ **Optimize** based on performance

---

_Last Updated: $(date)_
_Workflow Version: 1.0_
_Infrastructure: AWS Free Tier_
