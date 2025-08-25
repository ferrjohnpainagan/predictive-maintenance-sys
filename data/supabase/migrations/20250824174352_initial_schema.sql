-- Simple Schema for Quick Setup
-- Predictive Maintenance System - Essential Tables Only

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================
-- Engines Table
-- =====================================================
CREATE TABLE IF NOT EXISTS public.engines (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    unit_number INTEGER UNIQUE NOT NULL,
    aircraft_model VARCHAR(100) DEFAULT 'Boeing 737',
    engine_type VARCHAR(50) DEFAULT 'Turbofan',
    operational_status VARCHAR(20) DEFAULT 'active',
    current_rul INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- Sensor Data Table (Simplified)
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
    tra REAL,
    
    -- Key sensor readings (simplified set)
    s2 REAL,   -- Temperature
    s3 REAL,   -- Temperature  
    s4 REAL,   -- Temperature
    s7 REAL,   -- Pressure
    s8 REAL,   -- Fan speed
    s9 REAL,   -- Core speed
    s11 REAL,  -- Pressure
    s12 REAL,  -- Fuel flow ratio
    s13 REAL,  -- Corrected fan speed
    s14 REAL,  -- Corrected core speed
    s15 REAL,  -- Bypass ratio
    s17 REAL,  -- Coolant bleed
    s20 REAL,  -- Coolant bleed
    s21 REAL,  -- Coolant bleed
    
    -- Calculated fields
    rul INTEGER,
    health_score REAL,
    anomaly_score REAL,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (engine_id) REFERENCES public.engines(id) ON DELETE CASCADE
);

-- =====================================================
-- Indexes for Performance
-- =====================================================
CREATE INDEX IF NOT EXISTS idx_engines_unit_number ON public.engines(unit_number);
CREATE INDEX IF NOT EXISTS idx_sensor_data_engine_id ON public.sensor_data(engine_id);
CREATE INDEX IF NOT EXISTS idx_sensor_data_unit_number ON public.sensor_data(unit_number);
CREATE INDEX IF NOT EXISTS idx_sensor_data_timestamp ON public.sensor_data(timestamp DESC);

-- =====================================================
-- Alerts Table (Optional)
-- =====================================================
CREATE TABLE IF NOT EXISTS public.alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    engine_id UUID,
    unit_number INTEGER NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    details JSONB,
    is_acknowledged BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (engine_id) REFERENCES public.engines(id) ON DELETE CASCADE
);

-- =====================================================
-- Row Level Security (Enable but allow all for now)
-- =====================================================
ALTER TABLE public.engines ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sensor_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.alerts ENABLE ROW LEVEL SECURITY;

-- Create permissive policies for development
CREATE POLICY "Enable all operations for authenticated users" ON public.engines
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'service_role');

CREATE POLICY "Enable all operations for authenticated users" ON public.sensor_data
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'service_role');

CREATE POLICY "Enable all operations for authenticated users" ON public.alerts
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