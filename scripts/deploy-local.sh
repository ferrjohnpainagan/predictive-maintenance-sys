#!/bin/bash

# 🚀 Local Deployment Script for Predictive Maintenance System
# This script helps test deployment locally before using GitHub Actions

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
EC2_API_GATEWAY_IP="18.136.204.216"
EC2_ML_SERVICE_IP="13.215.159.154"
SSH_USER="ec2-user"
SSH_KEY="~/.ssh/predictive-maintenance-key"
DOCKER_REGISTRY="your-registry"  # Change this to your registry
IMAGE_TAG="test-$(date +%Y%m%d-%H%M%S)"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if SSH key exists
    if [ ! -f "$SSH_KEY" ]; then
        log_error "SSH key not found: $SSH_KEY"
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker is not running"
        exit 1
    fi
    
    # Check if docker-compose is available
    if ! command -v docker-compose > /dev/null; then
        log_error "docker-compose is not installed"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

test_ssh_connection() {
    log_info "Testing SSH connections..."
    
    # Test API Gateway connection
    if ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$SSH_USER@$EC2_API_GATEWAY_IP" "echo 'SSH to API Gateway successful'"; then
        log_success "SSH to API Gateway: OK"
    else
        log_error "SSH to API Gateway: FAILED"
        return 1
    fi
    
    # Test ML Service connection
    if ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$SSH_USER@$EC2_ML_SERVICE_IP" "echo 'SSH to ML Service successful'"; then
        log_success "SSH to ML Service: OK"
    else
        log_error "SSH to ML Service: FAILED"
        return 1
    fi
}

build_docker_images() {
    log_info "Building Docker images..."
    
    # Build API Gateway image
    log_info "Building API Gateway image..."
    if [ -d "backend/api-gateway" ]; then
        docker build -t "$DOCKER_REGISTRY/api-gateway:$IMAGE_TAG" backend/api-gateway/
        log_success "API Gateway image built: $DOCKER_REGISTRY/api-gateway:$IMAGE_TAG"
    else
        log_warning "API Gateway directory not found, skipping build"
    fi
    
    # Build ML Service image
    log_info "Building ML Service image..."
    if [ -d "mlops" ]; then
        docker build -t "$DOCKER_REGISTRY/ml-service:$IMAGE_TAG" mlops/
        log_success "ML Service image built: $DOCKER_REGISTRY/ml-service:$IMAGE_TAG"
    else
        log_warning "ML Service directory not found, skipping build"
    fi
}

deploy_to_api_gateway() {
    log_info "Deploying to API Gateway..."
    
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$SSH_USER@$EC2_API_GATEWAY_IP" << EOF
        # Set deployment variables
        DEPLOYMENT_ID="local-$(date +%Y%m%d-%H%M%S)"
        IMAGE_TAG="$IMAGE_TAG"
        DOCKER_REGISTRY="$DOCKER_REGISTRY"
        
        # Create deployment log
        echo "Starting local API Gateway deployment: \$DEPLOYMENT_ID at \$(date)" >> /var/log/deployment.log
        
        # Navigate to service directory
        cd /opt/api-gateway
        
        # Backup current configuration
        if [ -f docker-compose.yml ]; then
            cp docker-compose.yml docker-compose.yml.backup.\$(date +%Y%m%d_%H%M%S)
        fi
        
        # Update docker-compose.yml with new image
        cat > docker-compose.yml << 'COMPOSE_EOF'
        version: '3.8'
        services:
          api-gateway:
            image: \$DOCKER_REGISTRY/api-gateway:\$IMAGE_TAG
            ports: ["3000:3000"]
            environment: 
              - NODE_ENV=production
              - PORT=3000
              - DEPLOYMENT_ID=\$DEPLOYMENT_ID
            restart: unless-stopped
            healthcheck:
              test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
              interval: 30s
              timeout: 10s
              retries: 3
              start_period: 40s
        COMPOSE_EOF
        
        # Stop existing service
        docker-compose down || true
        
        # Start new service
        docker-compose up -d
        
        # Wait for service to be healthy
        echo "Waiting for API Gateway to be healthy..." >> /var/log/deployment.log
        for i in {1..30}; do
            if curl -f http://localhost:3000/health > /dev/null 2>&1; then
                echo "API Gateway is healthy after \$i attempts" >> /var/log/deployment.log
                break
            fi
            if [ \$i -eq 30 ]; then
                echo "ERROR: API Gateway failed health check after 30 attempts" >> /var/log/deployment.log
                exit 1
            fi
            sleep 2
        done
        
        # Verify deployment
        docker-compose ps
        curl -f http://localhost:3000/health
        
        echo "API Gateway deployment completed successfully: \$DEPLOYMENT_ID at \$(date)" >> /var/log/deployment.log
EOF
    
    if [ $? -eq 0 ]; then
        log_success "API Gateway deployment completed"
    else
        log_error "API Gateway deployment failed"
        return 1
    fi
}

deploy_to_ml_service() {
    log_info "Deploying to ML Service..."
    
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$SSH_USER@$EC2_ML_SERVICE_IP" << EOF
        # Set deployment variables
        DEPLOYMENT_ID="local-$(date +%Y%m%d-%H%M%S)"
        IMAGE_TAG="$IMAGE_TAG"
        DOCKER_REGISTRY="$DOCKER_REGISTRY"
        
        # Create deployment log
        echo "Starting local ML Service deployment: \$DEPLOYMENT_ID at \$(date)" >> /var/log/deployment.log
        
        # Navigate to service directory
        cd /opt/ml-service
        
        # Backup current configuration
        if [ -f docker-compose.yml ]; then
            cp docker-compose.yml docker-compose.yml.backup.\$(date +%Y%m%d_%H%M%S)
        fi
        
        # Update docker-compose.yml with new image
        cat > docker-compose.yml << 'COMPOSE_EOF'
        version: '3.8'
        services:
          ml-service:
            image: \$DOCKER_REGISTRY/ml-service:\$IMAGE_TAG
            ports: ["8000:8000", "50051:50051"]
            environment: 
              - ENVIRONMENT=production
              - PORT=8000
              - GRPC_PORT=50051
              - DEPLOYMENT_ID=\$DEPLOYMENT_ID
            restart: unless-stopped
            healthcheck:
              test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
              interval: 30s
              timeout: 10s
              retries: 3
              start_period: 40s
        COMPOSE_EOF
        
        # Stop existing service
        docker-compose down || true
        
        # Start new service
        docker-compose up -d
        
        # Wait for service to be healthy
        echo "Waiting for ML Service to be healthy..." >> /var/log/deployment.log
        for i in {1..30}; do
            if curl -f http://localhost:8000/health > /dev/null 2>&1; then
                echo "ML Service is healthy after \$i attempts" >> /var/log/deployment.log
                echo "ML Service is healthy after \$i attempts" >> /var/log/deployment.log
                break
            fi
            if [ \$i -eq 30 ]; then
                echo "ERROR: ML Service failed health check after 30 attempts" >> /var/log/deployment.log
                exit 1
            fi
            sleep 2
        done
        
        # Verify deployment
        docker-compose ps
        curl -f http://localhost:8000/health
        
        echo "ML Service deployment completed successfully: \$DEPLOYMENT_ID at \$(date)" >> /var/log/deployment.log
EOF
    
    if [ $? -eq 0 ]; then
        log_success "ML Service deployment completed"
    else
        log_error "ML Service deployment failed"
        return 1
    fi
}

verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check API Gateway
    log_info "Checking API Gateway..."
    if curl -f "http://$EC2_API_GATEWAY_IP:3000/health" > /dev/null 2>&1; then
        log_success "API Gateway: HEALTHY"
    else
        log_error "API Gateway: UNHEALTHY"
        return 1
    fi
    
    # Check ML Service
    log_info "Checking ML Service..."
    if curl -f "http://$EC2_ML_SERVICE_IP:8000/health" > /dev/null 2>&1; then
        log_success "ML Service: HEALTHY"
    else
        log_error "ML Service: UNHEALTHY"
        return 1
    fi
    
    log_success "All services deployed and healthy!"
}

show_deployment_info() {
    echo ""
    echo "🎉 Deployment Summary"
    echo "====================="
    echo "Deployment ID: local-$(date +%Y%m%d-%H%M%S)"
    echo "Image Tag: $IMAGE_TAG"
    echo "Docker Registry: $DOCKER_REGISTRY"
    echo ""
    echo "Service URLs:"
    echo "- API Gateway: http://$EC2_API_GATEWAY_IP:3000"
    echo "- ML Service: http://$EC2_ML_SERVICE_IP:8000"
    echo ""
    echo "Health Check Commands:"
    echo "- API Gateway: curl -f http://$EC2_API_GATEWAY_IP:3000/health"
    echo "- ML Service: curl -f http://$EC2_ML_SERVICE_IP:8000/health"
    echo ""
    echo "SSH Commands:"
    echo "- API Gateway: ssh -i $SSH_KEY $SSH_USER@$EC2_API_GATEWAY_IP"
    echo "- ML Service: ssh -i $SSH_KEY $SSH_USER@$EC2_ML_SERVICE_IP"
}

# Main execution
main() {
    echo "🚀 Predictive Maintenance System - Local Deployment"
    echo "=================================================="
    echo ""
    
    check_prerequisites
    test_ssh_connection
    build_docker_images
    deploy_to_api_gateway
    deploy_to_ml_service
    verify_deployment
    show_deployment_info
    
    log_success "Local deployment completed successfully!"
}

# Run main function
main "$@"
