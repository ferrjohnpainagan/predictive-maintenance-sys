#!/usr/bin/env python3
"""
Sample Configuration for Testing
Use this when Supabase is not available
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass  
class OfflineConfig:
    """Offline configuration for testing"""
    
    # Environment
    environment: str = "development"
    offline_mode: bool = True
    
    # File paths
    data_raw_path: str = "data/raw"
    data_processed_path: str = "data/processed"
    data_streaming_path: str = "data/streaming"
    logs_path: str = "logs"
    
    # Pipeline settings
    batch_size: int = 100
    max_memory_mb: int = 256
    num_engines: int = 3
    frequency_hz: float = 1.0
    
    # Quality thresholds
    min_quality_score: float = 0.6  # Relaxed for testing
    max_null_percentage: float = 0.1
    
    # Sensor ranges (relaxed for testing)
    sensor_ranges: Dict[str, tuple] = None
    
    def __post_init__(self):
        if self.sensor_ranges is None:
            self.sensor_ranges = {
                's2': (500, 800),    # Relaxed ranges
                's3': (1400, 1800),
                's4': (1200, 1600),
                's7': (2.0, 4.0),
                's8': (8000, 11000),
                's9': (7500, 10000),
                's11': (2.0, 4.0),
                's12': (0.3, 0.7),
                's13': (8000, 11000),
                's14': (7500, 10000),
                's15': (4.0, 7.0),
                's17': (0.03, 0.3),
                's20': (0.03, 0.3),
                's21': (0.03, 0.3)
            }

def get_offline_config():
    """Get offline configuration"""
    return OfflineConfig()
