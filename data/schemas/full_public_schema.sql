-- =====================================================
-- Predictive Maintenance System - Full Public Schema
-- Complete implementation using public schema for API access
-- =====================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================
-- Table: engines
-- Stores aircraft engine metadata
-- =====================================================
CREATE TABLE IF NOT EXISTS public.engines (
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

-- Indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_engines_unit_number ON public.engines(unit_number);
CREATE INDEX IF NOT EXISTS idx_engines_operational_status ON public.engines(operational_status);

-- =====================================================
-- Table: sensor_data
-- Stores time-series sensor readings 
-- Note: Removed partitioning for simplicity in free tier
-- =====================================================
CREATE TABLE IF NOT EXISTS public.sensor_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
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
    
    FOREIGN KEY (engine_id) REFERENCES public.engines(id) ON DELETE CASCADE
);

-- Indexes for optimized queries
CREATE INDEX IF NOT EXISTS idx_sensor_data_engine_id ON public.sensor_data(engine_id);
CREATE INDEX IF NOT EXISTS idx_sensor_data_unit_number ON public.sensor_data(unit_number);
CREATE INDEX IF NOT EXISTS idx_sensor_data_timestamp ON public.sensor_data(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_data_rul ON public.sensor_data(rul);

-- =====================================================
-- Table: predictions
-- Stores ML model predictions
-- =====================================================
CREATE TABLE IF NOT EXISTS public.predictions (
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
    
    FOREIGN KEY (engine_id) REFERENCES public.engines(id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_predictions_engine_id ON public.predictions(engine_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON public.predictions(prediction_timestamp DESC);

-- =====================================================
-- Table: maintenance_events
-- Stores maintenance history
-- =====================================================
CREATE TABLE IF NOT EXISTS public.maintenance_events (
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
    
    FOREIGN KEY (engine_id) REFERENCES public.engines(id) ON DELETE CASCADE,
    
    CONSTRAINT check_event_type CHECK (
        event_type IN ('scheduled', 'unscheduled', 'inspection', 'overhaul', 'failure')
    )
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_maintenance_events_engine_id ON public.maintenance_events(engine_id);
CREATE INDEX IF NOT EXISTS idx_maintenance_events_date ON public.maintenance_events(event_date DESC);

-- =====================================================
-- Table: alerts
-- Stores system alerts and notifications
-- =====================================================
CREATE TABLE IF NOT EXISTS public.alerts (
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
    
    FOREIGN KEY (engine_id) REFERENCES public.engines(id) ON DELETE CASCADE,
    
    CONSTRAINT check_severity CHECK (
        severity IN ('critical', 'high', 'medium', 'low', 'info')
    ),
    CONSTRAINT check_alert_type CHECK (
        alert_type IN ('rul_threshold', 'anomaly', 'sensor_failure', 'maintenance_due', 'prediction_update')
    )
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_alerts_engine_id ON public.alerts(engine_id);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON public.alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_is_acknowledged ON public.alerts(is_acknowledged);

-- =====================================================
-- Table: data_quality_metrics
-- Stores data quality metrics for monitoring
-- =====================================================
CREATE TABLE IF NOT EXISTS public.data_quality_metrics (
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
CREATE INDEX IF NOT EXISTS idx_data_quality_timestamp ON public.data_quality_metrics(check_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_data_quality_table ON public.data_quality_metrics(table_name);

-- =====================================================
-- Views for common queries
-- =====================================================

-- Current engine status view
CREATE OR REPLACE VIEW public.v_current_engine_status AS
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
FROM public.engines e
LEFT JOIN LATERAL (
    SELECT * FROM public.sensor_data
    WHERE engine_id = e.id
    ORDER BY timestamp DESC
    LIMIT 1
) sd ON true
LEFT JOIN LATERAL (
    SELECT * FROM public.predictions
    WHERE engine_id = e.id
    ORDER BY prediction_timestamp DESC
    LIMIT 1
) p ON true;

-- Recent alerts view
CREATE OR REPLACE VIEW public.v_recent_alerts AS
SELECT 
    a.*,
    e.aircraft_model,
    e.operational_status
FROM public.alerts a
JOIN public.engines e ON a.engine_id = e.id
WHERE a.created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
ORDER BY a.created_at DESC;

-- =====================================================
-- Functions for data management
-- =====================================================

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION public.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables with updated_at
CREATE TRIGGER update_engines_updated_at
    BEFORE UPDATE ON public.engines
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at();

CREATE TRIGGER update_maintenance_events_updated_at
    BEFORE UPDATE ON public.maintenance_events
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at();

-- =====================================================
-- Row Level Security (RLS) Policies
-- =====================================================

-- Enable RLS on all tables
ALTER TABLE public.engines ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sensor_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.maintenance_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.data_quality_metrics ENABLE ROW LEVEL SECURITY;

-- Create permissive policies for development
-- Note: In production, you should create more restrictive policies

CREATE POLICY "Enable all operations for authenticated users" ON public.engines
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'service_role');

CREATE POLICY "Enable all operations for authenticated users" ON public.sensor_data
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'service_role');

CREATE POLICY "Enable all operations for authenticated users" ON public.predictions
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'service_role');

CREATE POLICY "Enable all operations for authenticated users" ON public.maintenance_events
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'service_role');

CREATE POLICY "Enable all operations for authenticated users" ON public.alerts
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'service_role');

CREATE POLICY "Enable all operations for authenticated users" ON public.data_quality_metrics
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'service_role');

-- =====================================================
-- Sample Data (Optional - for testing)
-- =====================================================
INSERT INTO public.engines (unit_number, aircraft_model, operational_status, current_rul)
VALUES 
    (1, 'Boeing 737-800', 'active', 85),
    (2, 'Boeing 737-800', 'active', 92),
    (3, 'Boeing 737-MAX', 'active', 78)
ON CONFLICT (unit_number) DO NOTHING;