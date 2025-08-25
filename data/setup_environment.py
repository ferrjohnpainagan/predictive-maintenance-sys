#!/usr/bin/env python3
"""
Environment Setup Script
Creates necessary directories and configuration files for the data pipeline
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    directories = [
        'logs',
        'data/raw',
        'data/processed', 
        'data/streaming',
        'data/exports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

def create_env_template():
    """Create .env template file"""
    env_template = """# Predictive Maintenance System - Environment Variables
# Copy this file to .env and update with your actual values

# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key-here
SUPABASE_DB_PASSWORD=your-database-password

# AWS Configuration (Optional - for CloudWatch)
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=ap-southeast-1

# S3 Configuration (Optional)
S3_BUCKET_NAME=predictive-maintenance-free-tier-ml-data

# Environment
ENVIRONMENT=development

# Pipeline Configuration
DATA_PIPELINE_LOG_LEVEL=INFO
DATA_PIPELINE_BATCH_SIZE=500
DATA_PIPELINE_MAX_MEMORY_MB=512
"""
    
    env_file = Path('.env.template')
    env_file.write_text(env_template)
    logger.info(f"✅ Created environment template: {env_file}")
    
    # Check if .env already exists
    if not Path('.env').exists():
        logger.info("⚠️  Please copy .env.template to .env and update with your values")
        logger.info("   cp .env.template .env")
    else:
        logger.info("✅ .env file already exists")

def create_offline_mode_config():
    """Create configuration for offline mode (no Supabase)"""
    offline_config = """# Offline Mode Configuration
# This file enables running the pipeline without external dependencies

ENVIRONMENT=development
OFFLINE_MODE=true
USE_LOCAL_FILES=true
SKIP_DATABASE_CONNECTION=true
SKIP_CLOUDWATCH=true

# Local file storage
LOCAL_DATA_PATH=data/streaming
LOCAL_EXPORT_PATH=data/exports

# Pipeline settings optimized for local testing
DATA_PIPELINE_BATCH_SIZE=100
DATA_PIPELINE_MAX_MEMORY_MB=256
"""
    
    offline_file = Path('.env.offline')
    offline_file.write_text(offline_config)
    logger.info(f"✅ Created offline config: {offline_file}")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'pandas', 'numpy', 'psutil'
    ]
    
    optional_packages = [
        ('supabase', 'Database connectivity'),
        ('great_expectations', 'Data quality validation'),
        ('boto3', 'AWS CloudWatch integration'),
        ('pytest', 'Testing framework')
    ]
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            missing_required.append(package)
            logger.error(f"❌ {package} (REQUIRED)")
    
    # Check optional packages  
    for package, description in optional_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} - {description}")
        except ImportError:
            missing_optional.append((package, description))
            logger.warning(f"⚠️  {package} - {description} (OPTIONAL)")
    
    if missing_required:
        logger.error(f"Missing required packages: {missing_required}")
        logger.error("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        logger.info("Missing optional packages (pipeline will use fallbacks):")
        for package, description in missing_optional:
            logger.info(f"  pip install {package}  # {description}")
    
    return True

def create_sample_config():
    """Create sample configuration for testing"""
    sample_config = """#!/usr/bin/env python3
\"\"\"
Sample Configuration for Testing
Use this when Supabase is not available
\"\"\"

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass  
class OfflineConfig:
    \"\"\"Offline configuration for testing\"\"\"
    
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
    \"\"\"Get offline configuration\"\"\"
    return OfflineConfig()
"""
    
    config_file = Path('config/offline_config.py')
    config_file.write_text(sample_config)
    logger.info(f"✅ Created sample config: {config_file}")

def main():
    """Main setup function"""
    logger.info("=" * 60)
    logger.info("PREDICTIVE MAINTENANCE SYSTEM - ENVIRONMENT SETUP")
    logger.info("=" * 60)
    
    # Create directories
    logger.info("Creating directories...")
    create_directories()
    
    # Create environment templates
    logger.info("\nCreating environment configuration...")
    create_env_template()
    create_offline_mode_config()
    create_sample_config()
    
    # Check dependencies
    logger.info("\nChecking dependencies...")
    deps_ok = check_dependencies()
    
    # Final instructions
    logger.info("\n" + "=" * 60)
    logger.info("SETUP COMPLETE")
    logger.info("=" * 60)
    
    if deps_ok:
        logger.info("✅ All required dependencies are installed")
    else:
        logger.error("❌ Some required dependencies are missing")
        logger.error("   Please install missing packages and run setup again")
    
    logger.info("\nNext steps:")
    logger.info("1. Configure environment:")
    logger.info("   cp .env.template .env")
    logger.info("   # Edit .env with your Supabase credentials")
    logger.info("")
    logger.info("2. For offline testing (no Supabase):")
    logger.info("   cp .env.offline .env")
    logger.info("")
    logger.info("3. Test the pipeline:")
    logger.info("   python tests/test_integration.py")
    logger.info("")
    logger.info("4. Run development pipeline:")
    logger.info("   python scripts/continuous_pipeline.py --profile development --duration 30")
    
    return deps_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)