# Data Pipeline - Predictive Maintenance System

## Phase 3 Implementation Complete ✅

This data pipeline module provides a complete real-time streaming solution for predictive maintenance, including sensor simulation, continuous data processing, quality validation, and CloudWatch monitoring.

## Features Implemented

### Phase 2 Features ✅
- **Supabase Integration** - Database connection with connection pooling
- **Data Loading** - C-MAPSS dataset loader with memory optimization  
- **Data Quality Validation** - Great Expectations integration
- **Performance Testing** - Free tier constraint verification
- **Error Handling & Logging** - Comprehensive error handling

### Phase 3 Features ✅
- **Real-time Sensor Simulation** - Multi-engine sensor data generator
- **Streaming Data Processing** - Continuous data processing pipeline
- **Real-time Quality Validation** - Live data quality monitoring
- **CloudWatch Monitoring** - AWS metrics and alerting
- **GitHub Actions Automation** - CI/CD pipeline automation
- **Continuous Pipeline Orchestration** - Integrated real-time system

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Sensor          │───▶│ Streaming        │───▶│ Real-time       │
│ Simulator       │    │ Processor        │    │ Validator       │
│ (Multi-engine)  │    │ (Async)          │    │ (Quality Check) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Supabase        │    │ CloudWatch       │    │ GitHub Actions  │
│ Database        │    │ Monitoring       │    │ CI/CD           │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Project Structure

```
data/
├── config/
│   └── config.py                    # Configuration management
├── schemas/
│   └── supabase_schema.sql         # Database schema
├── scripts/
│   ├── sensor_simulator.py         # Real-time sensor data generator
│   ├── streaming_processor.py      # Streaming data processor
│   ├── realtime_validator.py       # Real-time quality validation
│   ├── cloudwatch_monitor.py       # CloudWatch monitoring
│   ├── continuous_pipeline.py      # Pipeline orchestrator
│   ├── supabase_connector.py       # Database connector
│   ├── data_ingestion.py          # Batch data ingestion
│   ├── data_quality_validator.py   # Data validation
│   └── performance_tester.py       # Performance testing
├── tests/
│   ├── test_integration.py         # Integration tests
│   ├── test_performance.py         # Performance tests
│   └── test_simple.py              # Unit tests
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Quick Start

### 1. Install Dependencies

```bash
cd data
pip install -r requirements.txt
```

### 2. Environment Configuration

Create `.env` file in project root:

```bash
# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key
SUPABASE_DB_PASSWORD=your_db_password

# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=ap-southeast-1

# Environment
ENVIRONMENT=development
```

### 3. Database Setup

```bash
# Run SQL schema in Supabase SQL Editor
cat schemas/supabase_schema.sql
```

### 4. Run Complete Pipeline

```bash
# Development mode (3 engines, 2Hz)
python scripts/continuous_pipeline.py --profile development --duration 60

# Demo mode with anomalies
python scripts/continuous_pipeline.py --profile demo --duration 30

# Production mode with CloudWatch
python scripts/continuous_pipeline.py --profile production --cloudwatch
```

## Usage Examples

### Real-time Sensor Simulation

```python
from scripts.sensor_simulator import SensorSimulator

# Create simulator for 5 engines at 2Hz
simulator = SensorSimulator(num_engines=5, frequency_hz=2.0)

# Register callback for readings
async def on_reading(reading):
    print(f"Engine {reading.unit_number}: RUL={reading.rul}")

simulator.register_callback('on_reading', on_reading)

# Run simulation
await simulator.start(duration_seconds=60)
```

### Streaming Data Processing

```python
from scripts.streaming_processor import StreamingProcessor

processor = StreamingProcessor()

# Start processing
await processor.start_processing()

# Process readings
result = await processor.process_reading(sensor_reading)

# Get metrics
metrics = processor.get_metrics()
print(f"Processed: {metrics['total_readings']}")
```

### Real-time Validation

```python
from scripts.realtime_validator import RealtimeValidator

validator = RealtimeValidator(window_size=100)

# Start validation service
await validator.start_validation_service()

# Validate reading
result = await validator.validate_reading(sensor_reading)

print(f"Quality Score: {result.quality_score}")
print(f"Issues: {result.issues}")
```

### CloudWatch Monitoring

```bash
# Setup CloudWatch monitoring
python scripts/cloudwatch_monitor.py --email your@email.com --test

# Send metrics programmatically
python -c "
from scripts.cloudwatch_monitor import CloudWatchMonitor
monitor = CloudWatchMonitor()
monitor.send_metric('DataQualityScore', 85.5, 'Percent')
"
```

## Pipeline Profiles

### Development Profile
- 3 engines, 2Hz frequency
- Small validation window (50 readings)
- Local logging only
- Ideal for: Development and testing

### Demo Profile  
- 2 engines, 3Hz frequency
- Fast validation (0.3s interval)
- Includes anomaly injection
- Ideal for: Demonstrations

### Production Profile
- 10 engines, 1Hz frequency  
- Large validation window (200 readings)
- CloudWatch integration
- Ideal for: Production deployment

### Testing Profile
- 2 engines, 5Hz frequency
- Minimal window (20 readings)
- High-speed testing
- Ideal for: Performance testing

## GitHub Actions Integration

The pipeline includes comprehensive CI/CD automation:

- **Code Quality**: Linting, formatting, testing
- **Integration Testing**: End-to-end pipeline tests
- **Data Quality Monitoring**: Automated quality checks
- **Production Deployment**: Automated data ingestion
- **CloudWatch Metrics**: Automated metric reporting

Workflows trigger on:
- Code pushes to main/feature branches
- Pull requests
- Scheduled runs (every 6 hours)
- Manual triggers

## CloudWatch Dashboard

The system creates a comprehensive CloudWatch dashboard with:

- **Data Quality Score** - Real-time quality metrics
- **Processing Throughput** - Records processed per second
- **Error Rates** - System error tracking
- **Memory Usage** - Resource utilization
- **Anomaly Detection** - Anomaly counts and trends

**Dashboard URL**: `https://console.aws.amazon.com/cloudwatch/home?region=ap-southeast-1#dashboards:name=PredictiveMaintenance-DataPipeline`

## Monitoring & Alerting

### Automated Alerts
- **Data Quality Low** (<70%) - Email notifications
- **High Anomaly Rate** (>10 anomalies) - Immediate alerts
- **Pipeline Failure** - Critical alerts
- **High Error Rate** (>10%) - Performance alerts
- **Memory Usage High** (>400MB) - Resource alerts

### Metrics Tracked
- Data quality scores
- Processing latency
- Throughput rates
- Error rates
- Memory usage
- Anomaly detection counts

## Performance Characteristics

### Throughput
- **Development**: ~6 readings/second (3 engines × 2Hz)
- **Production**: ~10 readings/second (10 engines × 1Hz)
- **Validation**: <50ms per reading
- **Database Insert**: ~100-500 records/batch

### Resource Usage
- **Memory**: <512MB (free tier optimized)
- **CPU**: Low utilization with async processing
- **Database**: Partitioned tables for efficiency
- **Storage**: Automatic cleanup of old data

## Troubleshooting

### Common Issues

**1. Connection Timeouts**
```bash
# Check Supabase credentials
echo $SUPABASE_URL
echo $SUPABASE_SERVICE_KEY
```

**2. High Memory Usage**  
```bash
# Reduce batch size or engine count
python scripts/continuous_pipeline.py --engines 2 --frequency 1.0
```

**3. CloudWatch Permissions**
```bash
# Verify AWS credentials
aws sts get-caller-identity
aws cloudwatch describe-alarms --max-items 1
```

**4. GitHub Actions Failures**
- Check secrets are configured in repository settings
- Verify AWS credentials have CloudWatch permissions
- Check Supabase database connection

### Debugging Commands

```bash
# Test individual components
python scripts/sensor_simulator.py --demo
python scripts/streaming_processor.py --demo  
python tests/test_integration.py

# View logs
tail -f logs/continuous_pipeline.log
tail -f logs/data_ingestion.log

# Check metrics
python -c "
from scripts.continuous_pipeline import ContinuousPipelineOrchestrator
import asyncio
# Run diagnostics
"
```

## API Reference

### SensorSimulator
- `SensorSimulator(num_engines, frequency_hz)` - Initialize simulator
- `start(duration_seconds)` - Start simulation  
- `inject_anomaly(unit_number, anomaly_type)` - Inject test anomalies
- `get_metrics()` - Get simulation metrics

### StreamingProcessor  
- `process_reading(reading)` - Process single reading
- `start_processing()` - Start background processing
- `get_metrics()` - Get processing metrics

### RealtimeValidator
- `validate_reading(reading)` - Validate single reading
- `start_validation_service()` - Start validation service
- `get_metrics()` - Get validation metrics

### CloudWatchMonitor
- `send_metric(name, value, unit)` - Send single metric
- `setup_monitoring(email)` - Setup alarms and topics
- `create_dashboard()` - Create CloudWatch dashboard

## Next Steps

Phase 3 is complete! Ready for:

1. **ML Model Integration** - Connect trained models for predictions
2. **API Gateway** - REST API for external access
3. **Frontend Dashboard** - Real-time monitoring UI
4. **Advanced Analytics** - Trend analysis and reporting
5. **Multi-tenant Support** - Support for multiple customers

## Support

- **Logs**: Check `logs/` directory for detailed execution logs
- **Metrics**: View CloudWatch dashboard for real-time metrics
- **Testing**: Run integration tests to verify functionality
- **Documentation**: Comprehensive inline code documentation