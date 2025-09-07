# AI Backend Deployment Guide

This guide covers the deployment and monitoring setup for the AI Backend service.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Monitoring Setup](#monitoring-setup)
5. [Health Checks](#health-checks)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+
- Git

### Environment Variables
Set the following environment variables before deployment:

```bash
export SUPABASE_URL="your-supabase-url"
export SUPABASE_SERVICE_KEY="your-service-key"
export SUPABASE_ANON_KEY="your-anon-key"
```

## Docker Deployment

### Development Deployment

```bash
# Clone the repository
git clone <repository-url>
cd ai-backend

# Start services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f ai-backend
```

### Production Deployment

```bash
# Use production configuration
docker-compose -f docker-compose.prod.yml up -d

# Or use the deployment script
./scripts/deployment/deploy.sh
```

### Deployment with Monitoring

```bash
# Start with monitoring stack
./scripts/deployment/deploy.sh monitoring

# Or manually
docker-compose -f docker-compose.prod.yml --profile monitoring up -d
```

## Kubernetes Deployment

### Prerequisites
- Kubernetes cluster 1.20+
- kubectl configured
- Helm 3.0+ (optional)

### Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace ai-backend

# Apply configurations
kubectl apply -f k8s/ -n ai-backend

# Check deployment status
kubectl get pods -n ai-backend
kubectl get services -n ai-backend
```

### Update Secrets

```bash
# Create secrets
kubectl create secret generic ai-backend-secrets \
  --from-literal=supabase-url="your-url" \
  --from-literal=supabase-service-key="your-key" \
  --from-literal=supabase-anon-key="your-anon-key" \
  -n ai-backend
```

## Monitoring Setup

### Prometheus Metrics

The AI Backend exposes metrics at `/metrics` endpoint:

- `ai_backend_health_status` - Overall health status (0/1)
- `ai_backend_cache_hit_rate` - Cache hit rate (0-1)
- `ai_backend_model_accuracy` - ML model accuracy by type
- `ai_backend_component_health` - Component health status
- `ai_backend_requests_per_minute` - Request rate

### Grafana Dashboards

1. Access Grafana at `http://localhost:3000`
2. Login with admin/admin (change password)
3. Import the dashboard from `monitoring/grafana/dashboards/`

### Log Aggregation

Logs are collected by Promtail and sent to Loki:

- Application logs: `/app/logs/ai_backend.log`
- Audit logs: `/app/logs/audit/ai_audit.log`
- Security logs: `/app/logs/audit/ai_security.log`

## Health Checks

### Available Endpoints

| Endpoint | Purpose | Use Case |
|----------|---------|----------|
| `/health` | Basic health check | Load balancer health check |
| `/health/detailed` | Comprehensive health status | Monitoring and debugging |
| `/health/live` | Liveness probe | Kubernetes liveness probe |
| `/health/ready` | Readiness probe | Kubernetes readiness probe |
| `/health/startup` | Startup probe | Kubernetes startup probe |
| `/metrics` | Prometheus metrics | Monitoring and alerting |

### Health Check Examples

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed

# Check specific component
curl -X POST http://localhost:8000/health/test/database
```

## CI/CD Pipeline

### GitHub Actions

The repository includes a complete CI/CD pipeline:

1. **Test Stage**: Runs unit tests and security scans
2. **Build Stage**: Builds and pushes Docker images
3. **Deploy Stage**: Deploys to staging/production

### Required Secrets

Set these secrets in your GitHub repository:

```
SUPABASE_URL
SUPABASE_SERVICE_KEY
SUPABASE_ANON_KEY
STAGING_SUPABASE_URL
STAGING_SUPABASE_SERVICE_KEY
STAGING_SUPABASE_ANON_KEY
PRODUCTION_SUPABASE_URL
PRODUCTION_SUPABASE_SERVICE_KEY
PRODUCTION_SUPABASE_ANON_KEY
SLACK_WEBHOOK (optional)
```

### Manual Deployment

```bash
# Deploy to staging
./scripts/deployment/deploy.sh

# Deploy to production
ENVIRONMENT=production ./scripts/deployment/deploy.sh
```

## Configuration Management

### Environment Files

- `.env.production` - Production configuration
- `.env.staging` - Staging configuration
- `.env` - Development configuration

### Feature Flags

Control features via environment variables:

```bash
FRAUD_DETECTION_ENABLED=true
PREDICTIVE_ANALYTICS_ENABLED=true
COMPLIANCE_CHECKING_ENABLED=true
NLP_INVOICE_ENABLED=true
```

## Scaling

### Horizontal Scaling

```bash
# Docker Compose
docker-compose up -d --scale ai-backend=3

# Kubernetes
kubectl scale deployment ai-backend --replicas=5 -n ai-backend
```

### Resource Limits

Configure resource limits in:
- `docker-compose.prod.yml` for Docker
- `k8s/deployment.yaml` for Kubernetes

## Security

### Network Security
- All services run in isolated Docker network
- TLS termination at load balancer
- Rate limiting enabled

### Data Security
- All data encrypted in transit and at rest
- Audit logging for all operations
- Data retention policies enforced

## Backup and Recovery

### Database Backups
Supabase handles database backups automatically.

### ML Model Backups
```bash
# Backup ML models
docker run --rm -v ai-backend_ml_models:/data -v $(pwd):/backup alpine tar czf /backup/ml_models_backup.tar.gz -C /data .

# Restore ML models
docker run --rm -v ai-backend_ml_models:/data -v $(pwd):/backup alpine tar xzf /backup/ml_models_backup.tar.gz -C /data
```

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
docker-compose logs ai-backend

# Check health
curl http://localhost:8000/health/detailed
```

#### Database Connection Issues
```bash
# Test database connection
curl -X POST http://localhost:8000/health/test/database

# Check environment variables
docker-compose exec ai-backend env | grep SUPABASE
```

#### High Memory Usage
```bash
# Check resource usage
docker stats

# Check ML model usage
curl http://localhost:8000/health/detailed | jq '.checks.ml_models'
```

#### Cache Issues
```bash
# Check Redis connection
docker-compose exec redis redis-cli ping

# Check cache stats
curl http://localhost:8000/health/detailed | jq '.checks.redis'
```

### Performance Tuning

#### Optimize ML Model Loading
```bash
# Reduce model cache size
export MODEL_CACHE_SIZE=3

# Increase model timeout
export MODEL_TIMEOUT_SECONDS=60
```

#### Database Optimization
```bash
# Increase connection pool
export DATABASE_MAX_CONNECTIONS=20

# Reduce timeout for faster failover
export DATABASE_TIMEOUT_SECONDS=5
```

### Monitoring and Alerting

#### Key Metrics to Monitor
- Response time (95th percentile < 2s)
- Error rate (< 1%)
- Memory usage (< 80%)
- Cache hit rate (> 80%)
- Model accuracy (> 70%)

#### Alert Thresholds
- Critical: Service down, database unreachable
- Warning: High response time, low cache hit rate
- Info: Model accuracy degradation

## Support

For deployment issues:
1. Check the troubleshooting section
2. Review service logs
3. Check health endpoints
4. Verify configuration

For additional help, contact the development team or create an issue in the repository.