# AI Backend Monitoring Guide

This guide covers the comprehensive monitoring and observability setup for the AI Backend service.

## Overview

The AI Backend includes a complete monitoring stack with:
- **Health Checks**: Multiple health check endpoints for different use cases
- **Metrics Collection**: Prometheus metrics for performance monitoring
- **Log Aggregation**: Centralized logging with Loki and Promtail
- **Visualization**: Grafana dashboards for metrics and logs
- **Alerting**: Prometheus alerting rules for proactive monitoring

## Health Check Endpoints

### Available Endpoints

| Endpoint | Purpose | Response Format | Use Case |
|----------|---------|-----------------|----------|
| `/health` | Basic health status | JSON | Load balancer health checks |
| `/health/detailed` | Comprehensive health with all components | JSON | Monitoring dashboards |
| `/health/live` | Liveness probe | JSON | Kubernetes liveness probe |
| `/health/ready` | Readiness probe | JSON | Kubernetes readiness probe |
| `/health/startup` | Startup probe | JSON | Kubernetes startup probe |
| `/health/dependencies` | External dependencies status | JSON | Dependency monitoring |
| `/health/test/{component}` | Test specific component | JSON | Debugging and testing |
| `/metrics` | Prometheus metrics | Text | Metrics collection |

### Health Check Components

The health check system monitors:

1. **Database Connection**
   - Connection status and response time
   - Query execution performance
   - Connection pool status

2. **Redis Cache**
   - Connection status and response time
   - Memory usage and hit rates
   - Connected clients count

3. **ML Models**
   - Loaded models count and status
   - Model performance metrics
   - Memory usage by models

4. **External Services**
   - GST API connectivity (if enabled)
   - Third-party service status
   - Network connectivity

5. **System Resources**
   - CPU usage percentage
   - Memory usage and availability
   - Disk space usage

### Example Health Check Response

```json
{
  "status": "healthy",
  "timestamp": 1640995200.0,
  "version": "v1",
  "environment": "production",
  "checks": {
    "database": {
      "status": "healthy",
      "response_time_ms": 15.2,
      "message": "Database connection successful"
    },
    "redis": {
      "status": "healthy",
      "response_time_ms": 2.1,
      "message": "Redis connection successful",
      "info": {
        "used_memory": "45.2M",
        "connected_clients": 3,
        "uptime_in_seconds": 86400
      }
    },
    "ml_models": {
      "status": "healthy",
      "loaded_models": 4,
      "models": {
        "fraud_detection": "loaded",
        "predictive_analytics": "loaded"
      }
    }
  },
  "features": {
    "fraud_detection": true,
    "predictive_analytics": true,
    "compliance_checking": true,
    "nlp_invoice": true
  },
  "metrics": {
    "cache_hit_rate": 0.85,
    "avg_response_time_ms": 150,
    "requests_per_minute": 45,
    "error_rate": 0.02
  }
}
```

## Prometheus Metrics

### Available Metrics

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `ai_backend_health_status` | Gauge | Overall health status (0/1) | - |
| `ai_backend_cache_hit_rate` | Gauge | Cache hit rate (0-1) | - |
| `ai_backend_model_accuracy` | Gauge | ML model accuracy | `model` |
| `ai_backend_component_health` | Gauge | Component health status (0/1) | `component` |
| `ai_backend_component_response_time_ms` | Gauge | Component response time | `component` |
| `ai_backend_requests_per_minute` | Gauge | Request rate | - |
| `ai_backend_error_rate` | Gauge | Error rate (0-1) | - |
| `ai_backend_avg_response_time_ms` | Gauge | Average response time | - |

### Custom Business Metrics

The system also tracks business-specific metrics:

```
# Fraud detection metrics
fraud_alerts_total{business_id, alert_type}
fraud_detection_accuracy{model_version}

# Predictive analytics metrics
prediction_requests_total{business_id, prediction_type}
prediction_accuracy{model_type}

# Compliance metrics
compliance_checks_total{business_id, check_type}
compliance_violations_total{business_id, violation_type}

# NLP processing metrics
nlp_requests_total{business_id, operation_type}
nlp_processing_time_seconds{operation_type}
```

## Grafana Dashboards

### Main Dashboard Panels

1. **Service Health Overview**
   - Service uptime status
   - Overall health score
   - Component status grid

2. **Performance Metrics**
   - Request rate and response time
   - Error rate trends
   - Cache performance

3. **Resource Usage**
   - CPU and memory usage
   - Database connection pool
   - Redis memory usage

4. **AI Model Performance**
   - Model accuracy trends
   - Prediction request rates
   - Model loading status

5. **Business Intelligence Metrics**
   - Fraud detection rates
   - Compliance check results
   - NLP processing volume

### Dashboard Access

- **URL**: `http://localhost:3000` (when monitoring is enabled)
- **Default Login**: admin/admin (change on first login)
- **Dashboard Location**: AI Backend folder

## Log Aggregation

### Log Types

1. **Application Logs** (`/app/logs/ai_backend.log`)
   - General application events
   - Request/response logging
   - Error messages and stack traces

2. **Audit Logs** (`/app/logs/audit/ai_audit.log`)
   - AI processing operations
   - Data access events
   - Security-related activities

3. **Security Logs** (`/app/logs/audit/ai_security.log`)
   - Authentication events
   - Security violations
   - Suspicious activities

4. **Performance Logs** (`/app/logs/performance.log`)
   - Response time metrics
   - Resource usage data
   - Performance bottlenecks

### Log Format

Logs are structured in JSON format for easy parsing:

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "logger": "app.services.fraud_detection",
  "message": "Fraud analysis completed",
  "business_id": "uuid-here",
  "processing_time_ms": 150,
  "results": {
    "alerts_generated": 2,
    "confidence_score": 0.85
  }
}
```

### Loki Queries

Common log queries for troubleshooting:

```logql
# All errors in the last hour
{job="ai-backend"} |= "ERROR" | json | __error__ = ""

# Fraud detection logs for specific business
{job="ai-backend"} | json | business_id="your-business-id" | operation_type="fraud_detection"

# High response time requests
{job="ai-backend"} | json | processing_time_ms > 1000

# Security violations
{job="ai-backend-audit", log_type="audit"} | json | event_type="SECURITY_VIOLATION"
```

## Alerting Rules

### Critical Alerts

1. **Service Down**
   - Trigger: Service unavailable for > 1 minute
   - Severity: Critical
   - Action: Immediate notification

2. **Database Connection Failed**
   - Trigger: Database unreachable for > 1 minute
   - Severity: Critical
   - Action: Immediate notification

3. **High Error Rate**
   - Trigger: Error rate > 10% for > 2 minutes
   - Severity: Warning
   - Action: Investigation required

### Performance Alerts

1. **High Response Time**
   - Trigger: 95th percentile > 2 seconds for > 5 minutes
   - Severity: Warning
   - Action: Performance investigation

2. **High Memory Usage**
   - Trigger: Memory usage > 1.5GB for > 5 minutes
   - Severity: Warning
   - Action: Resource optimization

3. **Low Cache Hit Rate**
   - Trigger: Cache hit rate < 50% for > 10 minutes
   - Severity: Warning
   - Action: Cache optimization

### Business Logic Alerts

1. **High Fraud Detection Rate**
   - Trigger: > 10 fraud alerts per hour
   - Severity: Warning
   - Action: Business review

2. **Model Accuracy Degradation**
   - Trigger: Model accuracy < 70% for > 15 minutes
   - Severity: Warning
   - Action: Model retraining

## Monitoring Setup

### Docker Compose Monitoring

Start the full monitoring stack:

```bash
# Start with monitoring enabled
docker compose -f docker-compose.prod.yml --profile monitoring up -d

# Or use the deployment script
./scripts/deployment/deploy.sh monitoring
```

### Kubernetes Monitoring

Deploy monitoring components:

```bash
# Deploy Prometheus
kubectl apply -f k8s/monitoring/prometheus.yaml

# Deploy Grafana
kubectl apply -f k8s/monitoring/grafana.yaml

# Deploy alert manager
kubectl apply -f k8s/monitoring/alertmanager.yaml
```

### Manual Monitoring Setup

1. **Install Prometheus**
   ```bash
   # Download and run Prometheus
   wget https://github.com/prometheus/prometheus/releases/download/v2.40.0/prometheus-2.40.0.linux-amd64.tar.gz
   tar xvfz prometheus-*.tar.gz
   cd prometheus-*
   ./prometheus --config.file=../monitoring/prometheus.yml
   ```

2. **Install Grafana**
   ```bash
   # Install Grafana
   sudo apt-get install -y software-properties-common
   sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
   sudo apt-get update
   sudo apt-get install grafana
   sudo systemctl start grafana-server
   ```

## Troubleshooting Monitoring

### Common Issues

1. **Metrics Not Appearing**
   - Check `/metrics` endpoint accessibility
   - Verify Prometheus configuration
   - Check network connectivity

2. **Grafana Dashboard Empty**
   - Verify Prometheus data source
   - Check query syntax
   - Confirm time range settings

3. **Alerts Not Firing**
   - Check alert rule syntax
   - Verify alert manager configuration
   - Check notification channels

### Monitoring Health Checks

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check Grafana health
curl http://localhost:3000/api/health

# Check Loki health
curl http://localhost:3100/ready

# Validate AI Backend metrics
curl http://localhost:8000/metrics
```

## Performance Optimization

### Monitoring Performance Impact

The monitoring system is designed to have minimal impact:

- Health checks are cached for 30 seconds
- Metrics collection uses efficient data structures
- Log rotation prevents disk space issues
- Background tasks run at low priority

### Optimization Tips

1. **Reduce Metric Cardinality**
   - Limit label values
   - Use appropriate metric types
   - Aggregate high-cardinality metrics

2. **Optimize Log Volume**
   - Use appropriate log levels
   - Implement log sampling for high-volume events
   - Set up log retention policies

3. **Cache Health Check Results**
   - Use cached results for frequent checks
   - Implement circuit breakers for failing components
   - Batch multiple checks together

## Security Considerations

### Monitoring Security

1. **Access Control**
   - Secure Grafana with authentication
   - Restrict Prometheus access
   - Use HTTPS for all monitoring endpoints

2. **Data Privacy**
   - Anonymize sensitive data in logs
   - Encrypt monitoring data in transit
   - Implement data retention policies

3. **Alert Security**
   - Secure notification channels
   - Validate alert sources
   - Monitor for monitoring system attacks

## Maintenance

### Regular Tasks

1. **Weekly**
   - Review dashboard performance
   - Check alert effectiveness
   - Update monitoring configurations

2. **Monthly**
   - Analyze monitoring data trends
   - Optimize slow queries
   - Update alert thresholds

3. **Quarterly**
   - Review monitoring architecture
   - Update monitoring tools
   - Conduct monitoring disaster recovery tests

### Backup and Recovery

1. **Configuration Backup**
   ```bash
   # Backup Grafana dashboards
   curl -H "Authorization: Bearer $GRAFANA_API_KEY" \
        http://localhost:3000/api/dashboards/db/ai-backend-dashboard

   # Backup Prometheus configuration
   cp monitoring/prometheus.yml monitoring/prometheus.yml.backup
   ```

2. **Data Backup**
   ```bash
   # Backup Prometheus data
   docker run --rm -v prometheus_data:/data -v $(pwd):/backup \
              alpine tar czf /backup/prometheus_backup.tar.gz -C /data .
   ```

This monitoring setup provides comprehensive observability for the AI Backend service, enabling proactive issue detection and performance optimization.