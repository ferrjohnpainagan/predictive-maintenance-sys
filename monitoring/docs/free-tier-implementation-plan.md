# Monitoring & Observability Free Tier Implementation Plan - Predictive Maintenance System

## Project Overview

Implementation plan for monitoring and observability using AWS Free Tier resources and open-source tools. This approach focuses on basic CloudWatch monitoring, simple logging, and lightweight observability tools to stay within free tier limits while maintaining system visibility.

## Technology Stack (Free Tier)

- **Metrics & Monitoring**: AWS CloudWatch (Free tier), Simple custom metrics
- **Logging**: AWS CloudWatch Logs (Free tier), Local log files
- **Health Checks**: Custom HTTP endpoints, Basic system monitoring
- **Dashboards**: CloudWatch Dashboards (Free tier), Simple HTML dashboards
- **Alerting**: Basic CloudWatch alarms, Email notifications
- **Infrastructure**: EC2 t2.micro instances

## Free Tier Considerations

- **CloudWatch**: 10 custom metrics, 5GB log ingestion, 3 dashboards
- **CloudWatch Alarms**: 10 alarms free
- **SNS**: 1,000 email notifications/month free
- **No premium APM tools**: Using basic monitoring only
- **Cost**: $0/month for 12 months

---

## Phase 1: Basic Observability Foundation

**Duration**: 3-4 days  
**Priority**: Critical

### Objectives

- Set up basic logging strategy
- Configure CloudWatch metrics collection
- Establish health check endpoints

### Tasks

1. **Logging Strategy (Free Tier)**

   ```python
   # shared/monitoring/logger.py
   import logging
   import json
   import sys
   from datetime import datetime
   from typing import Dict, Any
   import boto3

   class FreeTeamLogger:
       """Lightweight logger optimized for AWS free tier."""

       def __init__(self, service_name: str, enable_cloudwatch: bool = True):
           self.service_name = service_name
           self.enable_cloudwatch = enable_cloudwatch

           # Setup local logging
           self.local_logger = self._setup_local_logger()

           # Setup CloudWatch if enabled and within limits
           if self.enable_cloudwatch:
               try:
                   self.cloudwatch_client = boto3.client('logs')
                   self.log_group_name = f"/predictive-maintenance/{service_name}"
                   self._ensure_log_group_exists()
               except Exception as e:
                   self.local_logger.warning(f"CloudWatch setup failed: {e}")
                   self.enable_cloudwatch = False

       def _setup_local_logger(self) -> logging.Logger:
           """Setup local file and console logging."""

           logger = logging.getLogger(self.service_name)
           logger.setLevel(logging.INFO)

           # Console handler
           console_handler = logging.StreamHandler(sys.stdout)
           console_handler.setLevel(logging.INFO)

           # File handler (for local development)
           file_handler = logging.FileHandler(
               f"/var/log/{self.service_name}.log"
           )
           file_handler.setLevel(logging.INFO)

           # JSON formatter for structured logging
           formatter = logging.Formatter(
               '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
           )
           console_handler.setFormatter(formatter)
           file_handler.setFormatter(formatter)

           logger.addHandler(console_handler)
           logger.addHandler(file_handler)

           return logger

       def _ensure_log_group_exists(self):
           """Create CloudWatch log group if it doesn't exist."""
           try:
               self.cloudwatch_client.create_log_group(
                   logGroupName=self.log_group_name,
                   retentionInDays=7  # Free tier optimization
               )
           except self.cloudwatch_client.exceptions.ResourceAlreadyExistsException:
               pass

       def log(self, level: str, message: str, extra_data: Dict[str, Any] = None):
           """Log message to both local and CloudWatch."""

           # Structured log entry
           log_entry = {
               "timestamp": datetime.utcnow().isoformat(),
               "service": self.service_name,
               "level": level,
               "message": message,
               "extra": extra_data or {}
           }

           # Local logging
           log_message = json.dumps(log_entry)
           getattr(self.local_logger, level.lower())(log_message)

           # CloudWatch logging (if enabled and within limits)
           if self.enable_cloudwatch:
               try:
                   self._send_to_cloudwatch(log_entry)
               except Exception as e:
                   self.local_logger.warning(f"CloudWatch logging failed: {e}")

       def _send_to_cloudwatch(self, log_entry: Dict[str, Any]):
           """Send log entry to CloudWatch (with rate limiting)."""

           log_stream_name = f"{self.service_name}-{datetime.now().strftime('%Y-%m-%d')}"

           try:
               # Create log stream if needed
               try:
                   self.cloudwatch_client.create_log_stream(
                       logGroupName=self.log_group_name,
                       logStreamName=log_stream_name
                   )
               except self.cloudwatch_client.exceptions.ResourceAlreadyExistsException:
                   pass

               # Send log event
               self.cloudwatch_client.put_log_events(
                   logGroupName=self.log_group_name,
                   logStreamName=log_stream_name,
                   logEvents=[{
                       'timestamp': int(datetime.utcnow().timestamp() * 1000),
                       'message': json.dumps(log_entry)
                   }]
               )

           except Exception as e:
               # Fallback to local logging only
               self.local_logger.error(f"CloudWatch put_log_events failed: {e}")

       def info(self, message: str, extra_data: Dict[str, Any] = None):
           self.log("INFO", message, extra_data)

       def error(self, message: str, extra_data: Dict[str, Any] = None):
           self.log("ERROR", message, extra_data)

       def warning(self, message: str, extra_data: Dict[str, Any] = None):
           self.log("WARNING", message, extra_data)
   ```

2. **System Metrics Collector**

   ```python
   # shared/monitoring/metrics.py
   import psutil
   import time
   import boto3
   from typing import Dict, List, Any
   import threading

   class FreeTeamMetricsCollector:
       """Simple metrics collector for AWS free tier."""

       def __init__(self, service_name: str):
           self.service_name = service_name
           self.metrics_buffer = []
           self.max_buffer_size = 100  # Limit for free tier

           try:
               self.cloudwatch = boto3.client('cloudwatch')
               self.cloudwatch_enabled = True
           except Exception:
               self.cloudwatch_enabled = False

       def collect_system_metrics(self) -> Dict[str, float]:
           """Collect basic system metrics."""

           return {
               "cpu_percent": psutil.cpu_percent(interval=1),
               "memory_percent": psutil.virtual_memory().percent,
               "memory_used_mb": psutil.virtual_memory().used / 1024 / 1024,
               "disk_percent": psutil.disk_usage('/').percent,
               "disk_used_gb": psutil.disk_usage('/').used / 1024 / 1024 / 1024
           }

       def collect_application_metrics(self, custom_metrics: Dict[str, float] = None) -> Dict[str, float]:
           """Collect application-specific metrics."""

           metrics = {
               "timestamp": time.time(),
               "service": self.service_name
           }

           # Add system metrics
           metrics.update(self.collect_system_metrics())

           # Add custom metrics
           if custom_metrics:
               metrics.update(custom_metrics)

           return metrics

       def record_metric(self, metric_name: str, value: float, unit: str = "None"):
           """Record a custom metric."""

           metric_data = {
               "metric_name": metric_name,
               "value": value,
               "unit": unit,
               "timestamp": time.time(),
               "service": self.service_name
           }

           # Add to buffer
           self.metrics_buffer.append(metric_data)

           # Flush if buffer is full
           if len(self.metrics_buffer) >= self.max_buffer_size:
               self.flush_metrics()

       def flush_metrics(self):
           """Flush metrics to CloudWatch (respecting free tier limits)."""

           if not self.cloudwatch_enabled or not self.metrics_buffer:
               return

           try:
               # Group metrics by service for efficient sending
               metric_data = []

               for metric in self.metrics_buffer[:20]:  # Limit batch size
                   metric_data.append({
                       'MetricName': metric['metric_name'],
                       'Value': metric['value'],
                       'Unit': metric['unit'],
                       'Timestamp': metric['timestamp'],
                       'Dimensions': [
                           {
                               'Name': 'Service',
                               'Value': metric['service']
                           }
                       ]
                   })

               # Send to CloudWatch
               self.cloudwatch.put_metric_data(
                   Namespace='PredictiveMaintenance/FreeTier',
                   MetricData=metric_data
               )

               # Clear sent metrics
               self.metrics_buffer = self.metrics_buffer[20:]

           except Exception as e:
               print(f"Failed to send metrics to CloudWatch: {e}")
               # Keep metrics in buffer for retry

       def start_background_collection(self, interval_seconds: int = 60):
           """Start background metrics collection."""

           def collect_loop():
               while True:
                   try:
                       metrics = self.collect_application_metrics()

                       # Record core system metrics
                       self.record_metric("CPUUtilization", metrics["cpu_percent"], "Percent")
                       self.record_metric("MemoryUtilization", metrics["memory_percent"], "Percent")
                       self.record_metric("DiskUtilization", metrics["disk_percent"], "Percent")

                       time.sleep(interval_seconds)

                   except Exception as e:
                       print(f"Metrics collection error: {e}")
                       time.sleep(interval_seconds)

           thread = threading.Thread(target=collect_loop, daemon=True)
           thread.start()
   ```

3. **Health Check System**

   ```python
   # shared/monitoring/health.py
   import time
   import requests
   import psutil
   from typing import Dict, List, Any, Optional
   from dataclasses import dataclass

   @dataclass
   class HealthCheck:
       name: str
       check_function: callable
       timeout_seconds: int = 10
       critical: bool = True

   class HealthMonitor:
       """Simple health monitoring for free tier deployment."""

       def __init__(self, service_name: str):
           self.service_name = service_name
           self.checks: List[HealthCheck] = []
           self.last_check_results = {}

       def add_check(self, check: HealthCheck):
           """Add a health check."""
           self.checks.append(check)

       def add_http_check(self, name: str, url: str, timeout: int = 10, critical: bool = True):
           """Add HTTP endpoint health check."""

           def http_check():
               try:
                   response = requests.get(url, timeout=timeout)
                   return {
                       "status": "healthy" if response.status_code == 200 else "unhealthy",
                       "response_time_ms": response.elapsed.total_seconds() * 1000,
                       "status_code": response.status_code
                   }
               except Exception as e:
                   return {
                       "status": "unhealthy",
                       "error": str(e)
                   }

           self.add_check(HealthCheck(name, http_check, timeout, critical))

       def add_system_check(self, name: str = "system_resources"):
           """Add system resource health check."""

           def system_check():
               memory = psutil.virtual_memory()
               disk = psutil.disk_usage('/')
               cpu = psutil.cpu_percent(interval=1)

               # Determine health based on thresholds
               status = "healthy"
               warnings = []

               if memory.percent > 85:
                   status = "degraded"
                   warnings.append(f"High memory usage: {memory.percent:.1f}%")

               if disk.percent > 90:
                   status = "degraded"
                   warnings.append(f"High disk usage: {disk.percent:.1f}%")

               if cpu > 90:
                   status = "degraded"
                   warnings.append(f"High CPU usage: {cpu:.1f}%")

               if memory.percent > 95 or disk.percent > 95 or cpu > 95:
                   status = "unhealthy"

               return {
                   "status": status,
                   "memory_percent": memory.percent,
                   "disk_percent": disk.percent,
                   "cpu_percent": cpu,
                   "warnings": warnings
               }

           self.add_check(HealthCheck(name, system_check, 5, True))

       def run_checks(self) -> Dict[str, Any]:
           """Run all health checks and return results."""

           start_time = time.time()
           results = {
               "service": self.service_name,
               "timestamp": start_time,
               "checks": {},
               "overall_status": "healthy"
           }

           critical_failures = 0
           total_checks = len(self.checks)

           for check in self.checks:
               check_start = time.time()

               try:
                   check_result = check.check_function()
                   check_duration = (time.time() - check_start) * 1000

                   result = {
                       "status": check_result.get("status", "unknown"),
                       "duration_ms": check_duration,
                       "critical": check.critical,
                       "details": check_result
                   }

                   results["checks"][check.name] = result

                   # Track critical failures
                   if check.critical and check_result.get("status") != "healthy":
                       critical_failures += 1

               except Exception as e:
                   results["checks"][check.name] = {
                       "status": "error",
                       "duration_ms": (time.time() - check_start) * 1000,
                       "critical": check.critical,
                       "error": str(e)
                   }

                   if check.critical:
                       critical_failures += 1

           # Determine overall status
           if critical_failures > 0:
               results["overall_status"] = "unhealthy"
           elif critical_failures == 0 and total_checks > 0:
               # Check for any degraded services
               degraded_checks = [
                   r for r in results["checks"].values()
                   if r.get("status") == "degraded"
               ]
               if degraded_checks:
                   results["overall_status"] = "degraded"

           results["total_duration_ms"] = (time.time() - start_time) * 1000
           results["summary"] = {
               "total_checks": total_checks,
               "critical_failures": critical_failures,
               "healthy_checks": sum(1 for r in results["checks"].values() if r.get("status") == "healthy")
           }

           self.last_check_results = results
           return results

       def get_simple_status(self) -> Dict[str, str]:
           """Get simplified health status for lightweight endpoints."""

           if not self.last_check_results:
               self.run_checks()

           return {
               "status": self.last_check_results.get("overall_status", "unknown"),
               "service": self.service_name,
               "timestamp": time.time()
           }
   ```

### Deliverables

- [ ] Logging system configured for free tier
- [ ] Basic metrics collection implemented
- [ ] Health check framework established

---

## Phase 2: Application Performance Monitoring (Free Tier)

**Duration**: 4-5 days  
**Priority**: High

### Objectives

- Implement application-level monitoring
- Track API performance and errors
- Monitor resource usage patterns

### Tasks

1. **API Performance Monitoring**

   ```python
   # shared/monitoring/api_monitor.py
   import time
   import functools
   from typing import Dict, Any, Optional
   from flask import request, g
   from fastapi import Request

   class APIMonitor:
       """Simple API performance monitoring for free tier."""

       def __init__(self, metrics_collector):
           self.metrics_collector = metrics_collector
           self.request_count = 0
           self.error_count = 0
           self.response_times = []

       def track_request(self, method: str, endpoint: str, status_code: int,
                        response_time_ms: float, error: Optional[str] = None):
           """Track API request metrics."""

           self.request_count += 1
           self.response_times.append(response_time_ms)

           # Track errors
           if status_code >= 400:
               self.error_count += 1

           # Record metrics
           self.metrics_collector.record_metric(
               f"api_request_duration_ms",
               response_time_ms,
               "Milliseconds"
           )

           self.metrics_collector.record_metric(
               f"api_request_count",
               1,
               "Count"
           )

           if status_code >= 400:
               self.metrics_collector.record_metric(
                   f"api_error_count",
                   1,
                   "Count"
               )

       def get_performance_summary(self) -> Dict[str, Any]:
           """Get API performance summary."""

           if not self.response_times:
               return {"status": "no_data"}

           response_times = self.response_times[-100:]  # Last 100 requests

           return {
               "total_requests": self.request_count,
               "error_count": self.error_count,
               "error_rate": self.error_count / max(self.request_count, 1),
               "avg_response_time_ms": sum(response_times) / len(response_times),
               "max_response_time_ms": max(response_times),
               "min_response_time_ms": min(response_times),
               "p95_response_time_ms": sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0
           }

       def flask_middleware(self, app):
           """Flask middleware for automatic request tracking."""

           @app.before_request
           def before_request():
               g.start_time = time.time()

           @app.after_request
           def after_request(response):
               if hasattr(g, 'start_time'):
                   response_time = (time.time() - g.start_time) * 1000

                   self.track_request(
                       method=request.method,
                       endpoint=request.endpoint or 'unknown',
                       status_code=response.status_code,
                       response_time_ms=response_time
                   )

               return response

           return app

       def fastapi_middleware(self):
           """FastAPI middleware for automatic request tracking."""

           async def middleware(request: Request, call_next):
               start_time = time.time()

               response = await call_next(request)

               response_time = (time.time() - start_time) * 1000

               self.track_request(
                   method=request.method,
                   endpoint=str(request.url.path),
                   status_code=response.status_code,
                   response_time_ms=response_time
               )

               return response

           return middleware
   ```

2. **ML Model Performance Monitoring**

   ```python
   # shared/monitoring/ml_monitor.py
   import time
   import numpy as np
   from typing import Dict, List, Any, Optional
   from collections import deque

   class MLModelMonitor:
       """Monitor ML model performance for free tier."""

       def __init__(self, model_name: str, metrics_collector):
           self.model_name = model_name
           self.metrics_collector = metrics_collector

           # Prediction tracking
           self.prediction_times = deque(maxlen=1000)  # Last 1000 predictions
           self.prediction_count = 0
           self.error_count = 0

           # Model drift detection (simple)
           self.input_stats = deque(maxlen=100)  # Simple baseline

       def track_prediction(self,
                           input_data: np.ndarray,
                           prediction: float,
                           actual: Optional[float] = None,
                           prediction_time_ms: float = None,
                           error: Optional[str] = None):
           """Track ML prediction metrics."""

           self.prediction_count += 1

           if error:
               self.error_count += 1
               self.metrics_collector.record_metric(
                   f"ml_prediction_error_count",
                   1,
                   "Count"
               )
           else:
               # Track prediction time
               if prediction_time_ms:
                   self.prediction_times.append(prediction_time_ms)
                   self.metrics_collector.record_metric(
                       f"ml_prediction_latency_ms",
                       prediction_time_ms,
                       "Milliseconds"
                   )

               # Track prediction value
               self.metrics_collector.record_metric(
                   f"ml_prediction_value",
                   prediction,
                   "None"
               )

               # Simple input statistics for drift detection
               if input_data is not None:
                   input_mean = np.mean(input_data)
                   input_std = np.std(input_data)

                   self.input_stats.append({
                       'mean': input_mean,
                       'std': input_std,
                       'timestamp': time.time()
                   })

               # Track accuracy if actual value is provided
               if actual is not None:
                   mae = abs(prediction - actual)
                   self.metrics_collector.record_metric(
                       f"ml_prediction_mae",
                       mae,
                       "None"
                   )

           self.metrics_collector.record_metric(
               f"ml_prediction_count",
               1,
               "Count"
           )

       def get_model_health(self) -> Dict[str, Any]:
           """Get model health summary."""

           health_status = {
               "model_name": self.model_name,
               "total_predictions": self.prediction_count,
               "error_count": self.error_count,
               "error_rate": self.error_count / max(self.prediction_count, 1),
               "status": "healthy"
           }

           # Prediction latency analysis
           if self.prediction_times:
               times = list(self.prediction_times)
               health_status.update({
                   "avg_latency_ms": sum(times) / len(times),
                   "max_latency_ms": max(times),
                   "p95_latency_ms": sorted(times)[int(len(times) * 0.95)] if times else 0
               })

               # Check for performance degradation
               if health_status["avg_latency_ms"] > 2000:  # 2 seconds
                   health_status["status"] = "degraded"
                   health_status["warnings"] = ["High prediction latency"]

           # Simple drift detection
           if len(self.input_stats) > 50:
               recent_stats = list(self.input_stats)[-10:]  # Last 10 predictions
               baseline_stats = list(self.input_stats)[:10]  # First 10 predictions

               recent_mean = np.mean([s['mean'] for s in recent_stats])
               baseline_mean = np.mean([s['mean'] for s in baseline_stats])

               # Simple drift check (>20% change in input mean)
               if abs(recent_mean - baseline_mean) / abs(baseline_mean) > 0.2:
                   health_status["status"] = "degraded"
                   health_status["warnings"] = health_status.get("warnings", []) + ["Possible input drift detected"]

           return health_status

       def get_prediction_stats(self) -> Dict[str, Any]:
           """Get detailed prediction statistics."""

           if not self.prediction_times:
               return {"status": "no_data"}

           times = list(self.prediction_times)

           return {
               "prediction_count": len(times),
               "latency_stats": {
                   "min_ms": min(times),
                   "max_ms": max(times),
                   "avg_ms": sum(times) / len(times),
                   "median_ms": sorted(times)[len(times) // 2],
                   "p95_ms": sorted(times)[int(len(times) * 0.95)]
               },
               "error_rate": self.error_count / max(self.prediction_count, 1),
               "recent_input_stats": list(self.input_stats)[-5:] if self.input_stats else []
           }
   ```

3. **Resource Usage Monitoring**

   ```python
   # shared/monitoring/resource_monitor.py
   import psutil
   import time
   import threading
   from typing import Dict, List, Any
   from collections import deque

   class ResourceMonitor:
       """Monitor system resources for free tier deployment."""

       def __init__(self, metrics_collector, collection_interval: int = 60):
           self.metrics_collector = metrics_collector
           self.collection_interval = collection_interval

           # Resource history (limited for memory efficiency)
           self.cpu_history = deque(maxlen=100)
           self.memory_history = deque(maxlen=100)
           self.disk_history = deque(maxlen=100)

           self.monitoring = False
           self.monitor_thread = None

       def collect_resources(self) -> Dict[str, float]:
           """Collect current resource usage."""

           # CPU usage
           cpu_percent = psutil.cpu_percent(interval=1)

           # Memory usage
           memory = psutil.virtual_memory()
           memory_percent = memory.percent
           memory_used_mb = memory.used / 1024 / 1024

           # Disk usage
           disk = psutil.disk_usage('/')
           disk_percent = disk.percent
           disk_used_gb = disk.used / 1024 / 1024 / 1024

           # Network I/O (if available)
           try:
               network = psutil.net_io_counters()
               network_sent_mb = network.bytes_sent / 1024 / 1024
               network_recv_mb = network.bytes_recv / 1024 / 1024
           except Exception:
               network_sent_mb = 0
               network_recv_mb = 0

           timestamp = time.time()

           resource_data = {
               "timestamp": timestamp,
               "cpu_percent": cpu_percent,
               "memory_percent": memory_percent,
               "memory_used_mb": memory_used_mb,
               "disk_percent": disk_percent,
               "disk_used_gb": disk_used_gb,
               "network_sent_mb": network_sent_mb,
               "network_recv_mb": network_recv_mb
           }

           # Store in history
           self.cpu_history.append((timestamp, cpu_percent))
           self.memory_history.append((timestamp, memory_percent))
           self.disk_history.append((timestamp, disk_percent))

           return resource_data

       def start_monitoring(self):
           """Start background resource monitoring."""

           if self.monitoring:
               return

           self.monitoring = True

           def monitor_loop():
               while self.monitoring:
                   try:
                       resources = self.collect_resources()

                       # Send metrics to CloudWatch
                       self.metrics_collector.record_metric(
                           "CPUUtilization",
                           resources["cpu_percent"],
                           "Percent"
                       )
                       self.metrics_collector.record_metric(
                           "MemoryUtilization",
                           resources["memory_percent"],
                           "Percent"
                       )
                       self.metrics_collector.record_metric(
                           "DiskUtilization",
                           resources["disk_percent"],
                           "Percent"
                       )

                       time.sleep(self.collection_interval)

                   except Exception as e:
                       print(f"Resource monitoring error: {e}")
                       time.sleep(self.collection_interval)

           self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
           self.monitor_thread.start()

       def stop_monitoring(self):
           """Stop background resource monitoring."""
           self.monitoring = False

       def get_resource_summary(self) -> Dict[str, Any]:
           """Get resource usage summary."""

           current_resources = self.collect_resources()

           # Calculate trends if we have history
           cpu_trend = self._calculate_trend(self.cpu_history)
           memory_trend = self._calculate_trend(self.memory_history)
           disk_trend = self._calculate_trend(self.disk_history)

           return {
               "current": {
                   "cpu_percent": current_resources["cpu_percent"],
                   "memory_percent": current_resources["memory_percent"],
                   "memory_used_mb": current_resources["memory_used_mb"],
                   "disk_percent": current_resources["disk_percent"],
                   "disk_used_gb": current_resources["disk_used_gb"]
               },
               "trends": {
                   "cpu_trend": cpu_trend,
                   "memory_trend": memory_trend,
                   "disk_trend": disk_trend
               },
               "alerts": self._check_resource_alerts(current_resources),
               "history_points": len(self.cpu_history)
           }

       def _calculate_trend(self, history: deque) -> str:
           """Calculate simple trend from historical data."""

           if len(history) < 10:
               return "insufficient_data"

           recent = [point[1] for point in list(history)[-5:]]
           older = [point[1] for point in list(history)[-10:-5]]

           recent_avg = sum(recent) / len(recent)
           older_avg = sum(older) / len(older)

           diff_percent = ((recent_avg - older_avg) / older_avg) * 100

           if diff_percent > 5:
               return "increasing"
           elif diff_percent < -5:
               return "decreasing"
           else:
               return "stable"

       def _check_resource_alerts(self, resources: Dict[str, float]) -> List[str]:
           """Check for resource usage alerts."""

           alerts = []

           # Check CPU
           if resources["cpu_percent"] > 80:
               alerts.append(f"High CPU usage: {resources['cpu_percent']:.1f}%")

           # Check Memory
           if resources["memory_percent"] > 85:
               alerts.append(f"High memory usage: {resources['memory_percent']:.1f}%")

           # Check Disk
           if resources["disk_percent"] > 90:
               alerts.append(f"High disk usage: {resources['disk_percent']:.1f}%")

           # Check for t2.micro specific limits
           if resources["memory_used_mb"] > 800:  # Close to 1GB limit
               alerts.append(f"Memory usage near t2.micro limit: {resources['memory_used_mb']:.0f}MB")

           return alerts
   ```

### Deliverables

- [ ] API performance monitoring implemented
- [ ] ML model monitoring configured
- [ ] Resource usage tracking established

---

## Phase 3: Alerting & Notifications (Free Tier)

**Duration**: 3-4 days  
**Priority**: Medium

### Objectives

- Set up basic CloudWatch alarms
- Configure email notifications
- Implement simple alert routing

### Tasks

1. **CloudWatch Alarms Setup**

   ```python
   # monitoring/alerting/cloudwatch_alarms.py
   import boto3
   from typing import List, Dict, Any

   class CloudWatchAlerting:
       """Setup CloudWatch alarms for free tier monitoring."""

       def __init__(self, sns_topic_arn: str):
           self.cloudwatch = boto3.client('cloudwatch')
           self.sns_topic_arn = sns_topic_arn

       def create_cpu_alarm(self, service_name: str, threshold: float = 80):
           """Create CPU utilization alarm."""

           alarm_name = f"{service_name}-high-cpu"

           self.cloudwatch.put_metric_alarm(
               AlarmName=alarm_name,
               ComparisonOperator='GreaterThanThreshold',
               EvaluationPeriods=2,
               MetricName='CPUUtilization',
               Namespace='PredictiveMaintenance/FreeTier',
               Period=300,  # 5 minutes
               Statistic='Average',
               Threshold=threshold,
               ActionsEnabled=True,
               AlarmActions=[self.sns_topic_arn],
               AlarmDescription=f'CPU utilization high for {service_name}',
               Dimensions=[
                   {
                       'Name': 'Service',
                       'Value': service_name
                   }
               ]
           )

       def create_memory_alarm(self, service_name: str, threshold: float = 85):
           """Create memory utilization alarm."""

           alarm_name = f"{service_name}-high-memory"

           self.cloudwatch.put_metric_alarm(
               AlarmName=alarm_name,
               ComparisonOperator='GreaterThanThreshold',
               EvaluationPeriods=2,
               MetricName='MemoryUtilization',
               Namespace='PredictiveMaintenance/FreeTier',
               Period=300,
               Statistic='Average',
               Threshold=threshold,
               ActionsEnabled=True,
               AlarmActions=[self.sns_topic_arn],
               AlarmDescription=f'Memory utilization high for {service_name}',
               Dimensions=[
                   {
                       'Name': 'Service',
                       'Value': service_name
                   }
               ]
           )

       def create_api_error_alarm(self, service_name: str, threshold: float = 10):
           """Create API error rate alarm."""

           alarm_name = f"{service_name}-high-error-rate"

           self.cloudwatch.put_metric_alarm(
               AlarmName=alarm_name,
               ComparisonOperator='GreaterThanThreshold',
               EvaluationPeriods=3,
               MetricName='api_error_count',
               Namespace='PredictiveMaintenance/FreeTier',
               Period=300,
               Statistic='Sum',
               Threshold=threshold,
               ActionsEnabled=True,
               AlarmActions=[self.sns_topic_arn],
               AlarmDescription=f'High error rate for {service_name}',
               Dimensions=[
                   {
                       'Name': 'Service',
                       'Value': service_name
                   }
               ]
           )

       def setup_all_alarms(self, services: List[str]):
           """Setup standard alarms for all services."""

           for service in services:
               try:
                   self.create_cpu_alarm(service)
                   self.create_memory_alarm(service)
                   self.create_api_error_alarm(service)
                   print(f"Created alarms for {service}")
               except Exception as e:
                   print(f"Failed to create alarms for {service}: {e}")
   ```

2. **Simple Alert Manager**

   ```python
   # monitoring/alerting/alert_manager.py
   import time
   import smtplib
   from email.mime.text import MimeText
   from email.mime.multipart import MimeMultipart
   from typing import Dict, List, Any, Optional
   from collections import deque
   import boto3

   class SimpleAlertManager:
       """Simple alert management for free tier."""

       def __init__(self, sns_topic_arn: str = None, email_config: Dict = None):
           self.sns_topic_arn = sns_topic_arn
           self.email_config = email_config

           # Alert deduplication
           self.recent_alerts = deque(maxlen=100)
           self.alert_cooldown = 300  # 5 minutes

           if sns_topic_arn:
               self.sns = boto3.client('sns')
           else:
               self.sns = None

       def send_alert(self,
                      alert_type: str,
                      service: str,
                      message: str,
                      severity: str = "warning",
                      metadata: Dict[str, Any] = None):
           """Send alert via available channels."""

           # Check for duplicate alerts
           alert_key = f"{alert_type}:{service}:{severity}"
           if self._is_duplicate_alert(alert_key):
               return

           alert_data = {
               "timestamp": time.time(),
               "alert_type": alert_type,
               "service": service,
               "message": message,
               "severity": severity,
               "metadata": metadata or {}
           }

           # Send via SNS if available
           if self.sns:
               self._send_sns_alert(alert_data)

           # Send via email if configured
           if self.email_config:
               self._send_email_alert(alert_data)

           # Log alert
           print(f"ALERT [{severity.upper()}] {service}: {message}")

           # Track alert for deduplication
           self.recent_alerts.append({
               "key": alert_key,
               "timestamp": time.time()
           })

       def _is_duplicate_alert(self, alert_key: str) -> bool:
           """Check if alert is a duplicate within cooldown period."""

           current_time = time.time()

           for alert in self.recent_alerts:
               if (alert["key"] == alert_key and
                   current_time - alert["timestamp"] < self.alert_cooldown):
                   return True

           return False

       def _send_sns_alert(self, alert_data: Dict[str, Any]):
           """Send alert via SNS."""

           try:
               message = f"""
   ALERT: {alert_data['severity'].upper()}
   Service: {alert_data['service']}
   Type: {alert_data['alert_type']}
   Message: {alert_data['message']}
   Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(alert_data['timestamp']))}

   Metadata: {alert_data['metadata']}
   """

               self.sns.publish(
                   TopicArn=self.sns_topic_arn,
                   Subject=f"[{alert_data['severity'].upper()}] {alert_data['service']} - {alert_data['alert_type']}",
                   Message=message
               )

           except Exception as e:
               print(f"Failed to send SNS alert: {e}")

       def _send_email_alert(self, alert_data: Dict[str, Any]):
           """Send alert via email (basic SMTP)."""

           try:
               msg = MimeMultipart()
               msg['From'] = self.email_config['from_email']
               msg['To'] = self.email_config['to_email']
               msg['Subject'] = f"[{alert_data['severity'].upper()}] {alert_data['service']} Alert"

               body = f"""
   Alert Details:
   - Service: {alert_data['service']}
   - Type: {alert_data['alert_type']}
   - Severity: {alert_data['severity']}
   - Message: {alert_data['message']}
   - Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(alert_data['timestamp']))}

   Additional Information:
   {alert_data['metadata']}

   This is an automated alert from the Predictive Maintenance System.
   """

               msg.attach(MimeText(body, 'plain'))

               # Send email
               server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
               server.starttls()
               server.login(self.email_config['username'], self.email_config['password'])
               text = msg.as_string()
               server.sendmail(self.email_config['from_email'], self.email_config['to_email'], text)
               server.quit()

           except Exception as e:
               print(f"Failed to send email alert: {e}")

       def alert_high_resource_usage(self, service: str, resource_type: str, usage_percent: float):
           """Convenience method for resource usage alerts."""

           severity = "warning" if usage_percent < 90 else "critical"

           self.send_alert(
               alert_type="high_resource_usage",
               service=service,
               message=f"High {resource_type} usage: {usage_percent:.1f}%",
               severity=severity,
               metadata={
                   "resource_type": resource_type,
                   "usage_percent": usage_percent
               }
           )

       def alert_service_down(self, service: str, error_details: str):
           """Convenience method for service down alerts."""

           self.send_alert(
               alert_type="service_down",
               service=service,
               message=f"Service is down or unresponsive",
               severity="critical",
               metadata={
                   "error_details": error_details
               }
           )
   ```

### Deliverables

- [ ] CloudWatch alarms configured
- [ ] Email notification system implemented
- [ ] Alert management and deduplication setup

---

## Phase 4: Simple Dashboards (Free Tier)

**Duration**: 3-4 days  
**Priority**: Low

### Objectives

- Create basic CloudWatch dashboards
- Implement simple HTML dashboard
- Display key metrics and health status

### Tasks

1. **CloudWatch Dashboard Setup**

   ```python
   # monitoring/dashboards/cloudwatch_dashboard.py
   import boto3
   import json
   from typing import List, Dict, Any

   class CloudWatchDashboard:
       """Create CloudWatch dashboard for free tier monitoring."""

       def __init__(self):
           self.cloudwatch = boto3.client('cloudwatch')

       def create_system_dashboard(self, services: List[str], dashboard_name: str = "PredictiveMaintenance-FreeTier"):
           """Create system overview dashboard."""

           widgets = []

           # CPU Utilization widget
           widgets.append({
               "type": "metric",
               "x": 0, "y": 0, "width": 12, "height": 6,
               "properties": {
                   "metrics": [
                       ["PredictiveMaintenance/FreeTier", "CPUUtilization", "Service", service]
                       for service in services
                   ],
                   "period": 300,
                   "stat": "Average",
                   "region": "us-east-1",
                   "title": "CPU Utilization by Service",
                   "yAxis": {
                       "left": {"min": 0, "max": 100}
                   }
               }
           })

           # Memory Utilization widget
           widgets.append({
               "type": "metric",
               "x": 12, "y": 0, "width": 12, "height": 6,
               "properties": {
                   "metrics": [
                       ["PredictiveMaintenance/FreeTier", "MemoryUtilization", "Service", service]
                       for service in services
                   ],
                   "period": 300,
                   "stat": "Average",
                   "region": "us-east-1",
                   "title": "Memory Utilization by Service",
                   "yAxis": {
                       "left": {"min": 0, "max": 100}
                   }
               }
           })

           # API Request Count widget
           widgets.append({
               "type": "metric",
               "x": 0, "y": 6, "width": 12, "height": 6,
               "properties": {
                   "metrics": [
                       ["PredictiveMaintenance/FreeTier", "api_request_count", "Service", service]
                       for service in services
                   ],
                   "period": 300,
                   "stat": "Sum",
                   "region": "us-east-1",
                   "title": "API Request Count"
               }
           })

           # Prediction Latency widget
           widgets.append({
               "type": "metric",
               "x": 12, "y": 6, "width": 12, "height": 6,
               "properties": {
                   "metrics": [
                       ["PredictiveMaintenance/FreeTier", "ml_prediction_latency_ms", "Service", "ml-service"]
                   ],
                   "period": 300,
                   "stat": "Average",
                   "region": "us-east-1",
                   "title": "ML Prediction Latency"
               }
           })

           dashboard_body = {
               "widgets": widgets
           }

           try:
               self.cloudwatch.put_dashboard(
                   DashboardName=dashboard_name,
                   DashboardBody=json.dumps(dashboard_body)
               )
               print(f"Dashboard '{dashboard_name}' created successfully")

           except Exception as e:
               print(f"Failed to create dashboard: {e}")
   ```

2. **Simple HTML Dashboard**

   ```python
   # monitoring/dashboards/html_dashboard.py
   from flask import Flask, render_template, jsonify
   import json
   import time
   from typing import Dict, Any

   class SimpleDashboard:
       """Simple HTML dashboard for free tier monitoring."""

       def __init__(self, health_monitor, metrics_collector, api_monitor, ml_monitor):
           self.app = Flask(__name__)
           self.health_monitor = health_monitor
           self.metrics_collector = metrics_collector
           self.api_monitor = api_monitor
           self.ml_monitor = ml_monitor

           self._setup_routes()

       def _setup_routes(self):
           @self.app.route('/')
           def dashboard():
               return render_template('dashboard.html')

           @self.app.route('/api/health')
           def api_health():
               return jsonify(self.health_monitor.run_checks())

           @self.app.route('/api/metrics')
           def api_metrics():
               system_metrics = self.metrics_collector.collect_application_metrics()
               api_stats = self.api_monitor.get_performance_summary()
               ml_stats = self.ml_monitor.get_model_health()

               return jsonify({
                   "timestamp": time.time(),
                   "system": system_metrics,
                   "api": api_stats,
                   "ml_model": ml_stats
               })

           @self.app.route('/api/status')
           def api_status():
               overall_health = self.health_monitor.get_simple_status()

               return jsonify({
                   "status": overall_health["status"],
                   "timestamp": overall_health["timestamp"],
                   "services": {
                       "api_gateway": "healthy",  # Get from actual health checks
                       "ml_service": "healthy",
                       "database": "healthy"
                   }
               })

       def run(self, host='0.0.0.0', port=8080):
           """Run the dashboard server."""
           self.app.run(host=host, port=port, debug=False)
   ```

3. **Dashboard Template**
   ```html
   <!-- monitoring/dashboards/templates/dashboard.html -->
   <!DOCTYPE html>
   <html lang="en">
     <head>
       <meta charset="UTF-8" />
       <meta name="viewport" content="width=device-width, initial-scale=1.0" />
       <title>Predictive Maintenance - System Dashboard</title>
       <style>
         body {
           font-family: Arial, sans-serif;
           margin: 0;
           padding: 20px;
           background-color: #f5f5f5;
         }
         .dashboard {
           display: grid;
           grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
           gap: 20px;
         }
         .card {
           background: white;
           border-radius: 8px;
           padding: 20px;
           box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
         }
         .metric {
           font-size: 24px;
           font-weight: bold;
           margin: 10px 0;
         }
         .status {
           padding: 5px 10px;
           border-radius: 4px;
           color: white;
         }
         .healthy {
           background-color: #28a745;
         }
         .degraded {
           background-color: #ffc107;
           color: #000;
         }
         .unhealthy {
           background-color: #dc3545;
         }
         .chart {
           height: 200px;
           background: #f8f9fa;
           margin: 10px 0;
           display: flex;
           align-items: center;
           justify-content: center;
         }
       </style>
     </head>
     <body>
       <h1>Predictive Maintenance System - Free Tier Dashboard</h1>

       <div class="dashboard">
         <!-- System Health Card -->
         <div class="card">
           <h3>System Health</h3>
           <div id="overall-status" class="status">Loading...</div>
           <div id="health-details"></div>
         </div>

         <!-- Resource Usage Card -->
         <div class="card">
           <h3>Resource Usage</h3>
           <div>CPU: <span id="cpu-usage" class="metric">--</span>%</div>
           <div>Memory: <span id="memory-usage" class="metric">--</span>%</div>
           <div>Disk: <span id="disk-usage" class="metric">--</span>%</div>
         </div>

         <!-- API Performance Card -->
         <div class="card">
           <h3>API Performance</h3>
           <div>
             Requests: <span id="request-count" class="metric">--</span>
           </div>
           <div>
             Avg Response: <span id="avg-response" class="metric">--</span>ms
           </div>
           <div>
             Error Rate: <span id="error-rate" class="metric">--</span>%
           </div>
         </div>

         <!-- ML Model Card -->
         <div class="card">
           <h3>ML Model</h3>
           <div>
             Predictions: <span id="prediction-count" class="metric">--</span>
           </div>
           <div>
             Avg Latency:
             <span id="prediction-latency" class="metric">--</span>ms
           </div>
           <div>
             Model Status: <span id="model-status" class="status">--</span>
           </div>
         </div>

         <!-- Service Status Card -->
         <div class="card">
           <h3>Service Status</h3>
           <div>
             API Gateway: <span id="api-status" class="status">--</span>
           </div>
           <div>ML Service: <span id="ml-status" class="status">--</span></div>
           <div>Database: <span id="db-status" class="status">--</span></div>
         </div>

         <!-- Recent Alerts Card -->
         <div class="card">
           <h3>Recent Alerts</h3>
           <div id="recent-alerts">No recent alerts</div>
         </div>
       </div>

       <script>
         // Auto-refresh dashboard data
         function updateDashboard() {
           // Fetch health status
           fetch("/api/health")
             .then((response) => response.json())
             .then((data) => {
               document.getElementById("overall-status").textContent =
                 data.overall_status
               document.getElementById("overall-status").className =
                 "status " + data.overall_status
             })

           // Fetch metrics
           fetch("/api/metrics")
             .then((response) => response.json())
             .then((data) => {
               // Update system metrics
               document.getElementById("cpu-usage").textContent =
                 data.system.cpu_percent.toFixed(1)
               document.getElementById("memory-usage").textContent =
                 data.system.memory_percent.toFixed(1)
               document.getElementById("disk-usage").textContent =
                 data.system.disk_percent.toFixed(1)

               // Update API metrics
               if (data.api.total_requests) {
                 document.getElementById("request-count").textContent =
                   data.api.total_requests
                 document.getElementById("avg-response").textContent =
                   data.api.avg_response_time_ms.toFixed(1)
                 document.getElementById("error-rate").textContent = (
                   data.api.error_rate * 100
                 ).toFixed(2)
               }

               // Update ML metrics
               if (data.ml_model.total_predictions) {
                 document.getElementById("prediction-count").textContent =
                   data.ml_model.total_predictions
                 document.getElementById("prediction-latency").textContent =
                   data.ml_model.avg_latency_ms
                     ? data.ml_model.avg_latency_ms.toFixed(1)
                     : "--"

                 const modelStatus = document.getElementById("model-status")
                 modelStatus.textContent = data.ml_model.status
                 modelStatus.className = "status " + data.ml_model.status
               }
             })

           // Fetch service status
           fetch("/api/status")
             .then((response) => response.json())
             .then((data) => {
               Object.keys(data.services).forEach((service) => {
                 const element = document.getElementById(
                   service.replace("_", "-") + "-status"
                 )
                 if (element) {
                   element.textContent = data.services[service]
                   element.className = "status " + data.services[service]
                 }
               })
             })
         }

         // Update dashboard every 30 seconds
         updateDashboard()
         setInterval(updateDashboard, 30000)
       </script>
     </body>
   </html>
   ```

### Deliverables

- [ ] CloudWatch dashboard configured
- [ ] Simple HTML dashboard implemented
- [ ] Real-time metrics display working

---

## Migration Strategy to Paid Services

### Phase 1: Enhanced Monitoring Preparation (Month 10-11)

1. **APM Integration Planning**

   - Evaluate DataDog, New Relic, or X-Ray
   - Plan distributed tracing implementation
   - Design advanced alerting rules

2. **Advanced Metrics Strategy**
   - Custom business metrics
   - SLA/SLO monitoring
   - Performance benchmarking

### Phase 2: Migration Execution (Month 12)

1. **Migrate to Advanced APM**

   - Deploy DataDog or similar
   - Configure distributed tracing
   - Set up advanced dashboards

2. **Enhanced Alerting**
   - PagerDuty integration
   - Slack notifications
   - Escalation policies

---

## Success Metrics (Free Tier)

### Monitoring Coverage

- **System Metrics**: CPU, Memory, Disk monitored
- **Application Metrics**: API performance, ML latency tracked
- **Alert Response**: < 5 minute alert delivery
- **Dashboard Availability**: > 95% uptime

### Operational Targets

- **Cost**: $0/month (within free tier limits)
- **Alert Volume**: < 10 alerts/day (noise reduction)
- **Dashboard Load Time**: < 3 seconds
- **Log Storage**: < 4GB/month (within free tier)

---

## Timeline Summary

| Phase                             | Duration | Dependencies |
| --------------------------------- | -------- | ------------ |
| Phase 1: Observability Foundation | 3-4 days | AWS Account  |
| Phase 2: Application Monitoring   | 4-5 days | Phase 1      |
| Phase 3: Alerting & Notifications | 3-4 days | Phase 2      |
| Phase 4: Simple Dashboards        | 3-4 days | Phase 3      |

**Total Duration**: 13-17 days (2-3 weeks)

---

## Next Steps

1. **Immediate**: Set up basic logging and health checks
2. **Week 1**: Implement metrics collection and resource monitoring
3. **Week 2**: Configure alerting and notifications
4. **Week 3**: Deploy dashboards and validate monitoring
5. **Month 6-10**: Monitor and optimize system
6. **Month 12**: Migrate to advanced APM tools
