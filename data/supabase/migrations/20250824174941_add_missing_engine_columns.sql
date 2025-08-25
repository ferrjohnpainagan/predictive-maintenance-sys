-- Add missing columns to engines table
ALTER TABLE public.engines 
ADD COLUMN IF NOT EXISTS manufacture_date DATE,
ADD COLUMN IF NOT EXISTS installation_date DATE,
ADD COLUMN IF NOT EXISTS last_maintenance_date TIMESTAMP;

-- Update the aircraft_model column size if needed
ALTER TABLE public.engines ALTER COLUMN aircraft_model TYPE VARCHAR(50);

-- Add missing tables that might not have been created
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

-- Add missing indexes
CREATE INDEX IF NOT EXISTS idx_predictions_engine_id ON public.predictions(engine_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON public.predictions(prediction_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_maintenance_events_engine_id ON public.maintenance_events(engine_id);
CREATE INDEX IF NOT EXISTS idx_maintenance_events_date ON public.maintenance_events(event_date DESC);
CREATE INDEX IF NOT EXISTS idx_data_quality_timestamp ON public.data_quality_metrics(check_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_data_quality_table ON public.data_quality_metrics(table_name);

-- Add missing columns to alerts table
ALTER TABLE public.alerts 
ADD COLUMN IF NOT EXISTS acknowledged_by VARCHAR(100),
ADD COLUMN IF NOT EXISTS acknowledged_at TIMESTAMP WITH TIME ZONE;