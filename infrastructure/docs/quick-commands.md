# 🚀 Quick Commands Reference Card

## Predictive Maintenance System - Essential Commands

---

## 🔑 **SSH Access (Most Used)**

```bash
# API Gateway
ssh -i ~/.ssh/predictive-maintenance-key ec2-user@18.136.204.216

# ML Service
ssh -i ~/.ssh/predictive-maintenance-key ec2-user@13.215.159.154
```

---

## 🐳 **Docker Daily Commands**

```bash
# Check service status
docker ps
docker-compose ps

# View logs
docker-compose logs --tail=50
docker-compose logs -f

# Restart services
docker-compose restart

# Stop/Start services
docker-compose down
docker-compose up -d
```

---

## 🔍 **Health Checks**

```bash
# API Gateway health
curl -f http://18.136.204.216:3000/health

# ML Service health
curl -f http://13.215.159.154:8000/health

# Check health logs
tail -f /var/log/health-check.log
```

---

## 📊 **System Monitoring**

```bash
# Resource usage
top
df -h
free -h

# Port status
netstat -tlnp | grep :3000  # API Gateway
netstat -tlnp | grep :8000  # ML Service
```

---

## 🚨 **Emergency Commands**

```bash
# Restart everything
docker-compose restart

# Full restart
docker-compose down && docker-compose up -d

# Check Docker status
sudo systemctl status docker
sudo systemctl restart docker
```

---

## 📁 **Key Directories**

```bash
# Service configs
cd /opt/api-gateway
cd /opt/ml-service

# Logs
tail -f /var/log/startup.log
tail -f /var/log/health-check.log
```

---

## 🔗 **Service URLs**

- **API Gateway**: http://18.136.204.216:3000
- **ML Service**: http://13.215.159.154:8000
- **Health**: Port 8080 on both instances

---

_Keep this card handy for daily operations!_
_Full documentation: `service-commands.md`_
