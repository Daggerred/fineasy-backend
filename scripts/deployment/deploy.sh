#!/bin/bash

# AI Backend Deployment Script
set -e

# Configuration
ENVIRONMENT=${1:-production}
COMPOSE_FILE="docker-compose.prod.yml"
PROJECT_NAME="ai-backend"

echo "🚀 Starting AI Backend deployment for environment: $ENVIRONMENT"

# Check if required environment variables are set
check_env_vars() {
    local required_vars=(
        "SUPABASE_URL"
        "SUPABASE_SERVICE_KEY" 
        "SUPABASE_ANON_KEY"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            echo "❌ Error: Required environment variable $var is not set"
            exit 1
        fi
    done
    
    echo "✅ Environment variables validated"
}

# Pre-deployment checks
pre_deployment_checks() {
    echo "🔍 Running pre-deployment checks..."
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo "❌ Error: Docker is not running"
        exit 1
    fi
    
    # Check if docker-compose file exists
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        echo "❌ Error: Docker compose file $COMPOSE_FILE not found"
        exit 1
    fi
    
    # Check if required directories exist
    mkdir -p logs/audit ml_models monitoring
    
    echo "✅ Pre-deployment checks passed"
}

# Build and deploy services
deploy_services() {
    echo "🏗️ Building and deploying services..."
    
    # Pull latest images
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" pull
    
    # Build services
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" build --no-cache
    
    # Deploy services
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d
    
    echo "✅ Services deployed successfully"
}

# Wait for services to be healthy
wait_for_health() {
    echo "⏳ Waiting for services to be healthy..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        echo "Attempt $attempt/$max_attempts: Checking service health..."
        
        # Check AI Backend health
        if curl -f -s http://localhost:8000/health > /dev/null; then
            echo "✅ AI Backend is healthy"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            echo "❌ Error: Services failed to become healthy within timeout"
            docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs
            exit 1
        fi
        
        sleep 10
        ((attempt++))
    done
}

# Post-deployment verification
post_deployment_verification() {
    echo "🔍 Running post-deployment verification..."
    
    # Test API endpoints
    local endpoints=(
        "http://localhost:8000/health"
        "http://localhost:8000/health/detailed"
        "http://localhost:8000/health/ready"
        "http://localhost:8000/metrics"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -f -s "$endpoint" > /dev/null; then
            echo "✅ $endpoint is responding"
        else
            echo "❌ $endpoint is not responding"
            exit 1
        fi
    done
    
    # Check if monitoring is enabled
    if docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" --profile monitoring ps | grep -q "Up"; then
        echo "✅ Monitoring services are running"
        
        # Test monitoring endpoints
        if curl -f -s http://localhost:9090 > /dev/null; then
            echo "✅ Prometheus is accessible"
        fi
        
        if curl -f -s http://localhost:3000 > /dev/null; then
            echo "✅ Grafana is accessible"
        fi
    fi
    
    echo "✅ Post-deployment verification completed"
}

# Cleanup old containers and images
cleanup() {
    echo "🧹 Cleaning up old containers and images..."
    
    # Remove old containers
    docker container prune -f
    
    # Remove old images
    docker image prune -f
    
    echo "✅ Cleanup completed"
}

# Show deployment status
show_status() {
    echo "📊 Deployment Status:"
    echo "===================="
    
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps
    
    echo ""
    echo "🔗 Service URLs:"
    echo "AI Backend: http://localhost:8000"
    echo "Health Check: http://localhost:8000/health"
    echo "API Docs: http://localhost:8000/docs"
    echo "Prometheus: http://localhost:9090 (if monitoring enabled)"
    echo "Grafana: http://localhost:3000 (if monitoring enabled)"
    echo ""
    echo "📝 Logs:"
    echo "View logs with: docker compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f"
}

# Main deployment flow
main() {
    echo "🎯 AI Backend Deployment Script"
    echo "==============================="
    
    check_env_vars
    pre_deployment_checks
    deploy_services
    wait_for_health
    post_deployment_verification
    cleanup
    show_status
    
    echo ""
    echo "🎉 Deployment completed successfully!"
    echo "The AI Backend is now running and ready to serve requests."
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        echo "🛑 Stopping AI Backend services..."
        docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down
        echo "✅ Services stopped"
        ;;
    "restart")
        echo "🔄 Restarting AI Backend services..."
        docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" restart
        wait_for_health
        echo "✅ Services restarted"
        ;;
    "logs")
        docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs -f
        ;;
    "status")
        show_status
        ;;
    "monitoring")
        echo "📊 Starting with monitoring enabled..."
        docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" --profile monitoring up -d
        wait_for_health
        show_status
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|logs|status|monitoring}"
        echo ""
        echo "Commands:"
        echo "  deploy     - Deploy the AI Backend (default)"
        echo "  stop       - Stop all services"
        echo "  restart    - Restart all services"
        echo "  logs       - Show service logs"
        echo "  status     - Show deployment status"
        echo "  monitoring - Deploy with monitoring stack"
        exit 1
        ;;
esac