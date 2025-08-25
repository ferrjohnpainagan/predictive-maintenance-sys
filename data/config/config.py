"""
Data Pipeline Configuration - Predictive Maintenance System
Free Tier Optimized Configuration
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return True
    except ImportError:
        # Fallback: manually load .env file
        try:
            import os
            from pathlib import Path
            env_file = Path('.env')
            if env_file.exists():
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and '=' in line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
                return True
        except Exception:
            pass
        return False

# Load environment variables
load_env_file()

# =====================================================
# Environment Configuration
# =====================================================

class Environment:
    """Environment configuration"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

# =====================================================
# Database Configuration
# =====================================================

@dataclass
class DatabaseConfig:
    """Supabase database configuration"""
    url: str
    key: str
    schema: str = "public"
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create config from environment variables"""
        return cls(
            url=os.getenv('SUPABASE_URL', ''),
            key=os.getenv('SUPABASE_SERVICE_KEY', ''),
            schema=os.getenv('SUPABASE_SCHEMA', 'public')
        )

# =====================================================
# S3 Configuration
# =====================================================

@dataclass
class S3Config:
    """S3 storage configuration"""
    bucket_name: str
    region: str
    access_key: str
    secret_key: str
    
    @classmethod
    def from_env(cls) -> 'S3Config':
        """Create config from environment variables"""
        return cls(
            bucket_name=os.getenv('S3_BUCKET_NAME', 'predictive-maintenance-free-tier-ml-data'),
            region=os.getenv('AWS_REGION', 'ap-southeast-1'),
            access_key=os.getenv('AWS_ACCESS_KEY_ID', ''),
            secret_key=os.getenv('AWS_SECRET_ACCESS_KEY', '')
        )

# =====================================================
# Data Processing Configuration
# =====================================================

@dataclass
class DataProcessingConfig:
    """Data processing configuration (free tier optimized)"""
    # Batch processing for memory efficiency
    batch_size: int = 500  # Records per batch (optimized for t2.micro)
    max_workers: int = 2   # Parallel workers (limited for free tier)
    
    # Memory management
    max_memory_mb: int = 512  # Max memory usage (t2.micro has 1GB)
    chunk_size: int = 1000    # Process data in chunks
    
    # Data quality thresholds
    min_quality_score: float = 0.8
    max_null_percentage: float = 0.05
    
    # Sensor value ranges (based on C-MAPSS data)
    sensor_ranges: Dict[str, tuple] = None
    
    def __post_init__(self):
        if self.sensor_ranges is None:
            self.sensor_ranges = {
                's2': (630, 650),    # T24 - Total temperature at LPC outlet (°R)
                's3': (1550, 1650),  # T30 - Total temperature at HPC outlet (°R)
                's4': (1350, 1450),  # T50 - Total temperature at LPT outlet (°R)
                's7': (2.5, 3.5),    # P30 - Total pressure at HPC outlet (psia)
                's8': (9000, 10000), # Nf - Physical fan speed (rpm)
                's9': (8000, 9500),  # Nc - Physical core speed (rpm)
                's11': (2.5, 3.5),   # Ps30 - Static pressure at HPC outlet (psia)
                's12': (0.4, 0.6),   # phi - Ratio of fuel flow to Ps30 (pps/psia)
                's13': (9000, 10000), # NRf - Corrected fan speed (rpm)
                's14': (8000, 9500),  # NRc - Corrected core speed (rpm)
                's15': (5.0, 6.0),    # BPR - Bypass Ratio
                's17': (0.05, 0.15),  # htBleed - HPT coolant bleed (lbm/s)
                's20': (0.05, 0.15),  # W31 - LPT coolant bleed (lbm/s)
                's21': (0.05, 0.15)   # W32 - HPT coolant bleed (lbm/s)
            }

# =====================================================
# C-MAPSS Dataset Configuration
# =====================================================

@dataclass
class CMAPSSConfig:
    """C-MAPSS dataset configuration"""
    # Dataset files
    train_file: str = "train_FD001.txt"
    test_file: str = "test_FD001.txt"
    rul_file: str = "RUL_FD001.txt"
    
    # Dataset characteristics
    num_engines: int = 100  # Limit for free tier testing
    max_cycles: int = 500   # Maximum operational cycles
    num_sensors: int = 14   # Number of sensors to use
    
    # Column mappings
    column_names: list = None
    
    # Data types for memory optimization
    dtypes: Dict[str, str] = None
    
    def __post_init__(self):
        if self.column_names is None:
            self.column_names = [
                'unit_number', 'time_cycles', 'altitude', 'mach_number', 'tra'
            ] + [f's{i}' for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
        
        if self.dtypes is None:
            self.dtypes = {
                'unit_number': 'int32',
                'time_cycles': 'int32',
                'altitude': 'float32',
                'mach_number': 'float32',
                'tra': 'float32'
            }
            # Add sensor dtypes
            for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]:
                self.dtypes[f's{i}'] = 'float32'

# =====================================================
# Logging Configuration
# =====================================================

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/data_pipeline.log"
    max_file_size_mb: int = 10
    backup_count: int = 5

# =====================================================
# Main Configuration Class
# =====================================================

@dataclass
class DataPipelineConfig:
    """Main configuration class for data pipeline"""
    environment: str
    database: DatabaseConfig
    s3: S3Config
    data_processing: DataProcessingConfig
    cmapss: CMAPSSConfig
    logging: LoggingConfig = None
    
    @classmethod
    def from_env(cls, environment: str = Environment.DEVELOPMENT) -> 'DataPipelineConfig':
        """Create complete config from environment variables"""
        return cls(
            environment=environment,
            database=DatabaseConfig.from_env(),
            s3=S3Config.from_env(),
            data_processing=DataProcessingConfig(),
            cmapss=CMAPSSConfig(),
            logging=LoggingConfig()
        )

# =====================================================
# Configuration Instances
# =====================================================

def get_config(environment: str = None) -> DataPipelineConfig:
    """Get configuration instance"""
    if environment is None:
        environment = os.getenv('ENVIRONMENT', Environment.DEVELOPMENT)
    
    config = DataPipelineConfig.from_env(environment)
    return config
