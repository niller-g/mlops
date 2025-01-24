# MLOps Monitoring Stack

This directory contains the monitoring setup for our ML model serving infrastructure. The stack includes:
- FastAPI service for model inference
- Prometheus for metrics collection
- Grafana for metrics visualization
- Locust for load testing

## Quick Start

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

## Services Overview

### 1. ML Model API (Port 8000)
- **What**: FastAPI service serving our ML model
- **Access**: http://localhost:8000
- **Key Endpoints**:
  - `/` - Health check
  - `/infer` - Model inference endpoint
  - `/metrics` - Prometheus metrics endpoint
- **Example Usage**:
```bash
# Test inference
curl -X POST "http://localhost:8000/infer?prompt=What%20are%20the%20symptoms%20of"

# Check metrics
curl http://localhost:8000/metrics
```

### 2. Prometheus (Port 9090)
- **What**: Metrics collection and storage
- **Access**: http://localhost:9090
- **Key Features**:
  - Query metrics using PromQL
  - View targets and their health
  - Monitor scrape status
- **Useful Pages**:
  - `/targets` - Check scrape targets
  - `/graph` - Query and graph metrics
- **Example Queries**:
```promql
# Average inference latency
rate(model_inference_latency_seconds_sum[5m]) / rate(model_inference_latency_seconds_count[5m])

# CPU Usage
cpu_usage_percent

# RAM Usage
ram_usage_bytes
```

### 3. Grafana (Port 3000)
- **What**: Metrics visualization and dashboarding
- **Access**: http://localhost:3000
- **Login**: admin/admin
- **Key Features**:
  - Pre-configured Prometheus datasource
  - Custom dashboards for model monitoring
  - System resource visualization
- **Key Pages**:
  - Home → Dashboards
  - Explore (for ad-hoc queries)
  - Configuration → Data Sources

### 4. Locust (Port 8089)
- **What**: Load testing tool
- **Access**: http://localhost:8089
- **Features**:
  - Web UI for test configuration
  - Real-time metrics
  - Test scenarios:
    1. Basic inference (weight: 3) - Tests standard inference with random prompts
    2. Custom length inference (weight: 2) - Tests inference with varying output lengths
    3. Rapid stress test (weight: 1) - Rapid-fire requests for stress testing
    4. Metrics endpoint check (weight: 1) - Monitors system health
- **Running Tests**:
  1. Start with Docker (recommended):
     ```bash
     docker-compose up -d locust
     ```
  2. Or run locally (development):
     ```bash
     locust -f locust_tests.py --host http://localhost:8000 --web-port 8090
     ```
  3. Open web UI (http://localhost:8089 for Docker, or port 8090 if running locally)
  4. Set number of users (e.g., 10)
  5. Set spawn rate (e.g., 1 user/second)
  6. Click "Start swarming"

The test scenarios will automatically run with the specified weights, simulating real user behavior:
- Most users (3/7) will make basic inference requests
- Some users (2/7) will request custom-length responses
- Fewer users (1/7) will make rapid requests
- Background health checks (1/7) will monitor the metrics endpoint

## Metrics Available

### Model Metrics
- `model_inference_latency_seconds`: Histogram of inference times
- `training_loss`: Current training loss (if training)
- `validation_loss`: Current validation loss (if training)

### System Metrics
- `cpu_usage_percent`: CPU utilization
- `ram_usage_bytes`: Memory usage
- `gpu_memory_used_bytes`: GPU memory (if available)

### Validation Metrics
- `data_validation_checks_total`: Count of validation checks
- `validation_checks`: Status of individual checks

## Common Tasks

### 1. Monitor Model Performance
1. Open Grafana (http://localhost:3000)
2. Navigate to the ML Model dashboard
3. View inference latency and request rates

### 2. Run Load Tests
1. Open Locust (http://localhost:8089)
2. Configure test parameters
3. Monitor in real-time:
   - Response times
   - Request rates
   - Error rates

### 3. Check System Health
1. Open Prometheus (http://localhost:9090)
2. Go to Status → Targets
3. Verify all endpoints are "Up"

### 4. Debug Issues
1. Check container logs:
```bash
# View logs for specific service
docker-compose logs api
docker-compose logs prometheus
docker-compose logs grafana

# Follow logs in real-time
docker-compose logs -f api
```

2. Check container status:
```bash
docker-compose ps
```

## Maintenance

### Restart Services
```bash
# Restart single service
docker-compose restart api

# Restart everything
docker-compose restart
```

### Update Configuration
1. Edit configuration files:
   - `prometheus/prometheus.yml` for Prometheus
   - `grafana/provisioning/` for Grafana
   - `locust_tests.py` for load tests
2. Restart affected service:
```bash
docker-compose restart <service_name>
```

### Clean Up
```bash
# Stop all services
docker-compose down

# Remove volumes too
docker-compose down -v
``` 