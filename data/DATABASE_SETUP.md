# Database Setup Guide - Predictive Maintenance System

Complete guide for setting up the database locally or using Supabase in production.

## 🏗️ Architecture Overview

The system supports two deployment modes:
1. **Production**: Supabase cloud database with full schema and migrations
2. **Local Development**: Docker-based Supabase local setup

## 📊 Database Schema

Complete schema includes:
- **engines**: Aircraft engine metadata with maintenance history
- **sensor_data**: Real-time sensor readings with 14 C-MAPSS sensors
- **predictions**: ML model outputs and failure predictions
- **maintenance_events**: Maintenance history and scheduling
- **alerts**: System notifications and alerts
- **data_quality_metrics**: Data monitoring and validation metrics

## 🌍 Option 1: Production Setup (Supabase Cloud)

### Prerequisites
- Supabase account and project
- Node.js (for Supabase CLI)
- Python 3.9+

### Step 1: Install Supabase CLI
```bash
npm install -g supabase
# or use npx for one-time usage
```

### Step 2: Configure Environment
Create/update `.env` file:
```bash
# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_DB_PASSWORD=your-database-password

# Pipeline Configuration
ENVIRONMENT=production
DATA_PIPELINE_LOG_LEVEL=INFO
DATA_PIPELINE_BATCH_SIZE=1000
DATA_PIPELINE_MAX_MEMORY_MB=1024
```

### Step 3: Run Database Migrations
```bash
# Option 1: Using our migration script (recommended)
python3 scripts/setup_supabase.py --setup

# Option 2: Using Supabase CLI directly
supabase link --project-ref your-project-id
supabase db push
```

### Step 4: Verify Setup
```bash
# Test connection and tables
python3 scripts/setup_supabase.py --test

# Expected output:
# ✅ Supabase connection successful
# ✅ All tables created and accessible
# ✅ CRUD operations working
# ✅ Database ready for production
```

### Step 5: Run Production Pipeline
```bash
# Full production pipeline
python3 scripts/continuous_pipeline.py --profile production --duration 300

# Monitor performance
tail -f logs/continuous_pipeline.log
```

## 🏠 Option 2: Local Development Setup

### Prerequisites
- Docker Desktop
- Supabase CLI
- Python 3.9+

### Step 1: Initialize Local Supabase
```bash
# Initialize Supabase project (if not already done)
supabase init

# Start local Supabase stack
supabase start
```

### Step 2: Apply Migrations
```bash
# Apply database schema
supabase db reset

# Verify local setup
supabase db diff
```

### Step 3: Configure Local Environment
Update `.env` for local development:
```bash
# Local Supabase Configuration
SUPABASE_URL=http://localhost:54321
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...  # from supabase start output
SUPABASE_DB_PASSWORD=postgres

# Development Configuration
ENVIRONMENT=development
DATA_PIPELINE_LOG_LEVEL=DEBUG
DATA_PIPELINE_BATCH_SIZE=500
DATA_PIPELINE_MAX_MEMORY_MB=512
```

### Step 4: Run Development Pipeline
```bash
# Development pipeline with shorter batches
python3 scripts/continuous_pipeline.py --profile development --duration 60

# Access local Supabase dashboard
open http://localhost:54323
```

## 🛠️ Database Management Commands

### Using our Setup Script
```bash
# Complete setup (production or local)
make setup
# or
python3 scripts/setup_supabase.py --setup

# Test connection only
make test
# or
python3 scripts/setup_supabase.py --test

# Run migrations only
make migrate
# or
python3 scripts/setup_supabase.py --migrate

# Reset database (development only)
make reset
# or
python3 scripts/setup_supabase.py --reset
```

### Using Supabase CLI Directly
```bash
# Check status
supabase status

# View logs
supabase logs

# Generate types (optional)
supabase gen types typescript --local

# Stop local stack
supabase stop
```

## 📁 File Structure

### Production Files (Active)
```
data/
├── schemas/
│   └── full_public_schema.sql      # Complete production schema
├── scripts/
│   ├── setup_supabase.py           # CLI-based database setup
│   ├── continuous_pipeline.py      # Full production pipeline
│   └── supabase_connector.py       # Database connection layer
├── supabase/
│   ├── config.toml                 # Supabase configuration
│   └── migrations/                 # Version-controlled schema changes
│       ├── 20250824174352_initial_schema.sql
│       └── 20250824174941_add_missing_engine_columns.sql
├── Makefile                        # Convenience commands
└── DATABASE_SETUP.md              # This file
```

### Deprecated Files (Reference Only)
```
deprecated_files/
├── simple_implementations/        # Simple pipeline without full DB
├── testing_scripts/              # Basic connection tests
├── schemas/                       # Old schema versions
└── README.md                      # Explanation of deprecated files
```

## 🔍 Testing the Setup

### Connection Test
```bash
# Basic connection test
python3 -c "
from scripts.supabase_connector import SupabaseConnector
from config.config import get_config
config = get_config()
connector = SupabaseConnector(config.database)
print('✅ Connection successful!' if connector.test_connection() else '❌ Connection failed')
"
```

### Full Integration Test
```bash
# Run comprehensive test
python3 tests/test_integration.py

# Expected output:
# ✅ Database connection working
# ✅ Schema tables accessible
# ✅ CRUD operations successful
# ✅ Pipeline components integrated
# ✅ Data quality validation working
```

## 📊 Performance Benchmarks

### Production (Supabase Cloud)
- **Throughput**: 2-10 readings/second
- **Latency**: 50-200ms per batch
- **Success Rate**: >95%
- **Concurrent Engines**: 10-100+

### Local Development
- **Throughput**: 5-20 readings/second  
- **Latency**: 10-50ms per batch
- **Success Rate**: >98%
- **Concurrent Engines**: 5-50

## 🚨 Troubleshooting

### Common Issues

#### "Supabase CLI not found"
```bash
npm install -g supabase
# or use npx: npx supabase --version
```

#### "Migration failed - policy already exists"
```bash
# Reset and retry
supabase db reset --linked
python3 scripts/setup_supabase.py --setup
```

#### "Connection timeout"
```bash
# Check environment variables
cat .env | grep SUPABASE_URL

# Test connection manually
curl -H "apikey: your-service-key" your-supabase-url/rest/v1/
```

#### "Table not found"
```bash
# Verify migrations applied
supabase db diff --linked

# Re-run migrations
python3 scripts/setup_supabase.py --migrate
```

### Debug Mode
```bash
# Enable debug logging
export DATA_PIPELINE_LOG_LEVEL=DEBUG
python3 scripts/continuous_pipeline.py --profile development --duration 30
```

## 🎯 Success Criteria

### ✅ Database Setup Complete When:
- [ ] Environment variables configured correctly
- [ ] Supabase CLI working (production) or Docker running (local)
- [ ] Database migrations applied successfully
- [ ] Connection test passes
- [ ] All 6 tables created and accessible
- [ ] RLS policies configured properly
- [ ] Sample data can be inserted and retrieved

### ✅ Pipeline Integration Complete When:
- [ ] Continuous pipeline runs without errors
- [ ] Data appears in database tables in real-time
- [ ] Success rate >90%
- [ ] Throughput >1 reading/second
- [ ] Memory usage stable (<500MB for development)

## 🚀 Next Steps

1. **Verify Real-time Data**: Check Supabase dashboard for incoming sensor data
2. **Monitor Performance**: Set up alerting for pipeline health
3. **Scale Configuration**: Adjust engines and frequency for your use case
4. **Deploy Production**: Move from local to cloud infrastructure
5. **Implement ML Models**: Use predictions table for failure forecasting

## 📞 Support

- **Documentation**: See `docs/` directory for detailed guides
- **Issues**: Check `deprecated_files/` for reference implementations
- **Performance**: Run `scripts/performance_tester.py` for benchmarks

---

**Status**: ✅ Production-ready database setup with full schema and migrations support