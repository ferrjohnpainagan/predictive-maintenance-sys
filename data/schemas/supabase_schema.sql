-- =====================================================
-- Predictive Maintenance System - Supabase Schema
-- Free Tier Optimized Database Design
-- =====================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- =====================================================
-- Schema: predictive_maintenance
-- =====================================================
CREATE SCHEMA IF NOT EXISTS predictive_maintenance;

-- =====================================================
-- Table: engines
-- Stores aircraft engine metadata
-- =====================================================
CREATE TABLE IF NOT EXISTS predictive_maintenance.engines (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    unit_number INTEGER UNIQUE NOT NULL,
    aircraft_model VARCHAR(50),
    engine_type VARCHAR(50) DEFAULT 'Turbofan',
    manufacture_date DATE,
    installation_date DATE,
    operational_status VARCHAR(20) DEFAULT 'active',
    current_rul INTEGER,
    last_maintenance_date TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT check_operational_status CHECK (
        operational_status IN ('active', 'maintenance', 'failed', 'retired')
    )
);

-- Index for faster queries
CREATE INDEX idx_engines_unit_number ON predictive_maintenance.engines(unit_number);
CREATE INDEX idx_engines_operational_status ON predictive_maintenance.engines(operational_status);

-- =====================================================
-- Table: sensor_data
-- Stores time-series sensor readings (partitioned by month)
-- =====================================================
CREATE TABLE IF NOT EXISTS predictive_maintenance.sensor_data (
    id UUID DEFAULT uuid_generate_v4(),
    engine_id UUID NOT NULL,
    unit_number INTEGER NOT NULL,
    time_cycles INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Operational settings
    altitude REAL,
    mach_number REAL,
    tra REAL,  -- Throttle resolver angle
    
    -- Sensor readings (14 key sensors from C-MAPSS)
    s2 REAL,   -- T24 - Total temperature at LPC outlet
    s3 REAL,   -- T30 - Total temperature at HPC outlet
    s4 REAL,   -- T50 - Total temperature at LPT outlet
    s7 REAL,   -- P30 - Total pressure at HPC outlet
    s8 REAL,   -- Nf - Physical fan speed
    s9 REAL,   -- Nc - Physical core speed
    s11 REAL,  -- Ps30 - Static pressure at HPC outlet
    s12 REAL,  -- phi - Ratio of fuel flow to Ps30
    s13 REAL,  -- NRf - Corrected fan speed
    s14 REAL,  -- NRc - Corrected core speed
    s15 REAL,  -- BPR - Bypass Ratio
    s17 REAL,  -- htBleed - HPT coolant bleed
    s20 REAL,  -- W31 - LPT coolant bleed
    s21 REAL,  -- W32 - HPT coolant bleed
    
    -- Calculated fields
    rul INTEGER,  -- Remaining Useful Life
    health_score REAL,  -- Engine health score (0-100)
    anomaly_score REAL,  -- Anomaly detection score
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (timestamp, engine_id),
    FOREIGN KEY (engine_id) REFERENCES predictive_maintenance.engines(id) ON DELETE CASCADE
) PARTITION BY RANGE (timestamp);

-- Create partitions for the last 6 months (free tier optimization)
CREATE TABLE IF NOT EXISTS predictive_maintenance.sensor_data_2024_08 
    PARTITION OF predictive_maintenance.sensor_data 
    FOR VALUES FROM ('2024-08-01') TO ('2024-09-01');

CREATE TABLE IF NOT EXISTS predictive_maintenance.sensor_data_2024_09 
    PARTITION OF predictive_maintenance.sensor_data 
    FOR VALUES FROM ('2024-09-01') TO ('2024-10-01');

CREATE TABLE IF NOT EXISTS predictive_maintenance.sensor_data_2024_10 
    PARTITION OF predictive_maintenance.sensor_data 
    FOR VALUES FROM ('2024-10-01') TO ('2024-11-01');

CREATE TABLE IF NOT EXISTS predictive_maintenance.sensor_data_2024_11 
    PARTITION OF predictive_maintenance.sensor_data 
    FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');

CREATE TABLE IF NOT EXISTS predictive_maintenance.sensor_data_2024_12 
    PARTITION OF predictive_maintenance.sensor_data 
    FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');

CREATE TABLE IF NOT EXISTS predictive_maintenance.sensor_data_2025_01 
    PARTITION OF predictive_maintenance.sensor_data 
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- Indexes for optimized queries
CREATE INDEX idx_sensor_data_engine_id ON predictive_maintenance.sensor_data(engine_id);
CREATE INDEX idx_sensor_data_unit_number ON predictive_maintenance.sensor_data(unit_number);
CREATE INDEX idx_sensor_data_timestamp ON predictive_maintenance.sensor_data(timestamp DESC);
CREATE INDEX idx_sensor_data_rul ON predictive_maintenance.sensor_data(rul);

-- =====================================================
-- Table: predictions
-- Stores ML model predictions
-- =====================================================
CREATE TABLE IF NOT EXISTS predictive_maintenance.predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    engine_id UUID NOT NULL,
    unit_number INTEGER NOT NULL,
    prediction_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    
    -- Predictions
    predicted_rul INTEGER NOT NULL,
    confidence_score REAL,
    failure_probability REAL,
    predicted_failure_date DATE,
    
    -- Model metadata
    features_used JSONB,
    model_parameters JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (engine_id) REFERENCES predictive_maintenance.engines(id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX idx_predictions_engine_id ON predictive_maintenance.predictions(engine_id);
CREATE INDEX idx_predictions_timestamp ON predictive_maintenance.predictions(prediction_timestamp DESC);

-- =====================================================
-- Table: maintenance_events
-- Stores maintenance history
-- =====================================================
CREATE TABLE IF NOT EXISTS predictive_maintenance.maintenance_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    engine_id UUID NOT NULL,
    unit_number INTEGER NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    event_date TIMESTAMP WITH TIME ZONE NOT NULL,
    description TEXT,
    cost DECIMAL(10, 2),
    downtime_hours INTEGER,
    technician_notes TEXT,
    parts_replaced JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (engine_id) REFERENCES predictive_maintenance.engines(id) ON DELETE CASCADE,
    
    CONSTRAINT check_event_type CHECK (
        event_type IN ('scheduled', 'unscheduled', 'inspection', 'overhaul', 'failure')
    )
);

-- Indexes
CREATE INDEX idx_maintenance_events_engine_id ON predictive_maintenance.maintenance_events(engine_id);
CREATE INDEX idx_maintenance_events_date ON predictive_maintenance.maintenance_events(event_date DESC);

-- =====================================================
-- Table: alerts
-- Stores system alerts and notifications
-- =====================================================
CREATE TABLE IF NOT EXISTS predictive_maintenance.alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    engine_id UUID NOT NULL,
    unit_number INTEGER NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    details JSONB,
    is_acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (engine_id) REFERENCES predictive_maintenance.engines(id) ON DELETE CASCADE,
    
    CONSTRAINT check_severity CHECK (
        severity IN ('critical', 'high', 'medium', 'low', 'info')
    ),
    CONSTRAINT check_alert_type CHECK (
        alert_type IN ('rul_threshold', 'anomaly', 'sensor_failure', 'maintenance_due', 'prediction_update')
    )
);

-- Indexes
CREATE INDEX idx_alerts_engine_id ON predictive_maintenance.alerts(engine_id);
CREATE INDEX idx_alerts_severity ON predictive_maintenance.alerts(severity);
CREATE INDEX idx_alerts_is_acknowledged ON predictive_maintenance.alerts(is_acknowledged);

-- =====================================================
-- Table: data_quality_metrics
-- Stores data quality metrics for monitoring
-- =====================================================
CREATE TABLE IF NOT EXISTS predictive_maintenance.data_quality_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    check_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    
    -- Metrics
    total_records INTEGER,
    null_count INTEGER,
    duplicate_count INTEGER,
    anomaly_count INTEGER,
    quality_score REAL,
    
    -- Detailed results
    check_results JSONB,
    issues_found TEXT[],
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_data_quality_timestamp ON predictive_maintenance.data_quality_metrics(check_timestamp DESC);
CREATE INDEX idx_data_quality_table ON predictive_maintenance.data_quality_metrics(table_name);

-- =====================================================
-- Views for common queries
-- =====================================================

-- Current engine status view
CREATE OR REPLACE VIEW predictive_maintenance.v_current_engine_status AS
SELECT 
    e.id,
    e.unit_number,
    e.aircraft_model,
    e.operational_status,
    e.current_rul,
    sd.timestamp as last_reading_time,
    sd.health_score,
    sd.anomaly_score,
    p.predicted_rul,
    p.confidence_score,
    p.predicted_failure_date
FROM predictive_maintenance.engines e
LEFT JOIN LATERAL (
    SELECT * FROM predictive_maintenance.sensor_data
    WHERE engine_id = e.id
    ORDER BY timestamp DESC
    LIMIT 1
) sd ON true
LEFT JOIN LATERAL (
    SELECT * FROM predictive_maintenance.predictions
    WHERE engine_id = e.id
    ORDER BY prediction_timestamp DESC
    LIMIT 1
) p ON true;

-- Recent alerts view
CREATE OR REPLACE VIEW predictive_maintenance.v_recent_alerts AS
SELECT 
    a.*,
    e.aircraft_model,
    e.operational_status
FROM predictive_maintenance.alerts a
JOIN predictive_maintenance.engines e ON a.engine_id = e.id
WHERE a.created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
ORDER BY a.created_at DESC;

-- =====================================================
-- Functions for data management
-- =====================================================

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION predictive_maintenance.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables with updated_at
CREATE TRIGGER update_engines_updated_at
    BEFORE UPDATE ON predictive_maintenance.engines
    FOR EACH ROW
    EXECUTE FUNCTION predictive_maintenance.update_updated_at();

CREATE TRIGGER update_maintenance_events_updated_at
    BEFORE UPDATE ON predictive_maintenance.maintenance_events
    FOR EACH ROW
    EXECUTE FUNCTION predictive_maintenance.update_updated_at();

-- Function to create monthly partitions automatically
CREATE OR REPLACE FUNCTION predictive_maintenance.create_monthly_partition()
RETURNS void AS $$
DECLARE
    partition_date DATE;
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    -- Get next month's date
    partition_date := DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month');
    partition_name := 'sensor_data_' || TO_CHAR(partition_date, 'YYYY_MM');
    start_date := partition_date;
    end_date := partition_date + INTERVAL '1 month';
    
    -- Check if partition already exists
    IF NOT EXISTS (
        SELECT 1 FROM pg_tables 
        WHERE schemaname = 'predictive_maintenance' 
        AND tablename = partition_name
    ) THEN
        -- Create partition
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS predictive_maintenance.%I PARTITION OF predictive_maintenance.sensor_data FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date
        );
        
        RAISE NOTICE 'Created partition: %', partition_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- Row Level Security (RLS) Policies
-- =====================================================

-- Enable RLS on sensitive tables
ALTER TABLE predictive_maintenance.engines ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictive_maintenance.sensor_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictive_maintenance.predictions ENABLE ROW LEVEL SECURITY;

-- Create policies (adjust based on your auth setup)
-- For now, allowing all authenticated users to read data
CREATE POLICY "Enable read access for all users" ON predictive_maintenance.engines
    FOR SELECT USING (true);

CREATE POLICY "Enable read access for all users" ON predictive_maintenance.sensor_data
    FOR SELECT USING (true);

CREATE POLICY "Enable read access for all users" ON predictive_maintenance.predictions
    FOR SELECT USING (true);

-- =====================================================
-- Grants for application user
-- =====================================================
-- Note: Adjust the username based on your Supabase setup
-- GRANT USAGE ON SCHEMA predictive_maintenance TO authenticated;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA predictive_maintenance TO authenticated;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA predictive_maintenance TO authenticated;