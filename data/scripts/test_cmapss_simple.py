"""
Simple Test Script for C-MAPSS Data Generation
Tests the core functionality without requiring database configuration
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.config import CMAPSSConfig

# =====================================================
# Simple Test Configuration
# =====================================================

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def generate_sample_cmapss_data():
    """Generate sample C-MAPSS data for testing"""
    logger = logging.getLogger(__name__)
    
    # Create test configuration
    config = CMAPSSConfig(
        num_engines=5,  # Small number for testing
        max_cycles=10,   # Small number for testing
        num_sensors=14
    )
    
    logger.info("Generating sample C-MAPSS data...")
    
    # Generate training data
    train_data = generate_training_data(config)
    logger.info(f"Generated {len(train_data)} training records")
    
    # Generate test data
    test_data = generate_test_data(config)
    logger.info(f"Generated {len(test_data)} test records")
    
    # Generate RUL data
    rul_data = generate_rul_data(config)
    logger.info(f"Generated RUL data for {len(rul_data)} engines")
    
    # Save data
    save_data(train_data, test_data, rul_data)
    
    # Test data validation
    validate_sample_data(train_data)
    
    return True

def generate_training_data(config):
    """Generate sample training data based on C-MAPSS characteristics"""
    try:
        data = []
        
        # Generate data for each engine
        for engine_num in range(1, config.num_engines + 1):
            # Random number of cycles for this engine (5-10)
            max_cycles = np.random.randint(5, config.max_cycles + 1)
            
            for cycle in range(1, max_cycles + 1):
                # Operational settings (realistic ranges)
                altitude = np.random.uniform(30000, 40000)  # ft
                mach_number = np.random.uniform(0.7, 0.9)   # Mach
                tra = np.random.uniform(80, 90)             # %
                
                # Sensor readings with realistic noise and degradation
                degradation_factor = 1 + (cycle * 0.001)  # Gradual degradation
                
                sensor_readings = []
                for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]:
                    # Base values for each sensor
                    base_values = {
                        2: 640, 3: 1600, 4: 1400, 7: 3.0, 8: 9500, 9: 8750,
                        11: 3.0, 12: 0.5, 13: 9500, 14: 8750, 15: 5.5,
                        17: 0.1, 20: 0.1, 21: 0.1
                    }
                    
                    base_value = base_values[i]
                    degraded_value = base_value * degradation_factor
                    
                    # Add realistic noise (±5%)
                    noise = np.random.normal(0, degraded_value * 0.05)
                    sensor_value = max(0, degraded_value + noise)
                    sensor_readings.append(sensor_value)
                
                # Create record
                record = [engine_num, cycle, altitude, mach_number, tra] + sensor_readings
                data.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=config.column_names)
        
        # Add RUL and timestamps
        df = add_rul_labels(df)
        df = add_timestamps(df)
        
        return df
        
    except Exception as e:
        logging.error(f"Error generating training data: {e}")
        raise

def generate_test_data(config):
    """Generate sample test data"""
    try:
        # Similar to training data but with fewer engines
        test_engines = min(3, config.num_engines // 2)
        
        data = []
        for engine_num in range(1, test_engines + 1):
            max_cycles = np.random.randint(3, 8)
            
            for cycle in range(1, max_cycles + 1):
                # Similar logic to training data
                altitude = np.random.uniform(30000, 40000)
                mach_number = np.random.uniform(0.7, 0.9)
                tra = np.random.uniform(80, 90)
                
                degradation_factor = 1 + (cycle * 0.001)
                
                sensor_readings = []
                for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]:
                    base_values = {
                        2: 640, 3: 1600, 4: 1400, 7: 3.0, 8: 9500, 9: 8750,
                        11: 3.0, 12: 0.5, 13: 9500, 14: 8750, 15: 5.5,
                        17: 0.1, 20: 0.1, 21: 0.1
                    }
                    
                    base_value = base_values[i]
                    degraded_value = base_value * degradation_factor
                    noise = np.random.normal(0, degraded_value * 0.05)
                    sensor_value = max(0, degraded_value + noise)
                    sensor_readings.append(sensor_value)
                
                record = [engine_num, cycle, altitude, mach_number, tra] + sensor_readings
                data.append(record)
        
        df = pd.DataFrame(data, columns=config.column_names)
        df = add_rul_labels(df)
        df = add_timestamps(df)
        
        return df
        
    except Exception as e:
        logging.error(f"Error generating test data: {e}")
        raise

def generate_rul_data(config):
    """Generate sample RUL data"""
    try:
        data = []
        
        for engine_num in range(1, config.num_engines + 1):
            # Random RUL value (0-125 cycles)
            rul = np.random.randint(0, 126)
            data.append([engine_num, rul])
        
        df = pd.DataFrame(data, columns=['unit_number', 'rul'])
        return df
        
    except Exception as e:
        logging.error(f"Error generating RUL data: {e}")
        raise

def add_rul_labels(df, max_rul=125):
    """Calculate RUL for each engine cycle"""
    try:
        # Group by engine and find max cycles
        max_cycles = df.groupby('unit_number')['time_cycles'].max()
        
        def calculate_rul(row):
            max_cycle = max_cycles[row['unit_number']]
            rul = max_cycle - row['time_cycles']
            return min(rul, max_rul)
        
        df['rul'] = df.apply(calculate_rul, axis=1)
        return df
        
    except Exception as e:
        logging.error(f"Error adding RUL labels: {e}")
        raise

def add_timestamps(df):
    """Add realistic timestamps based on operational cycles"""
    try:
        # Assume each cycle represents 1 day of operation
        base_date = datetime(2020, 1, 1)
        
        def calculate_timestamp(row):
            return base_date + timedelta(days=int(row['time_cycles']))
        
        df['timestamp'] = df.apply(calculate_timestamp, axis=1)
        return df
        
    except Exception as e:
        logging.error(f"Error adding timestamps: {e}")
        raise

def save_data(train_data, test_data, rul_data):
    """Save generated data to files"""
    logger = logging.getLogger(__name__)
    
    # Save training data
    train_file = "data/raw/train_FD001.txt"
    train_data.to_csv(train_file, sep=' ', header=False, index=False)
    logger.info(f"Training data saved to {train_file}")
    
    # Save test data
    test_file = "data/raw/test_FD001.txt"
    test_data.to_csv(test_file, sep=' ', header=False, index=False)
    logger.info(f"Test data saved to {test_file}")
    
    # Save RUL data
    rul_file = "data/raw/RUL_FD001.txt"
    rul_data.to_csv(rul_file, sep=' ', header=False, index=False)
    logger.info(f"RUL data saved to {rul_file}")
    
    # Save processed data as parquet
    output_file = "data/processed/training_data_processed.parquet"
    train_data.to_parquet(output_file, compression='snappy')
    logger.info(f"Processed data saved to {output_file}")

def validate_sample_data(df):
    """Validate the generated sample data"""
    logger = logging.getLogger(__name__)
    
    logger.info("=== Data Validation Results ===")
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Number of engines: {df['unit_number'].nunique()}")
    logger.info(f"Data columns: {list(df.columns)}")
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        logger.info("✅ No null values found")
    else:
        logger.warning(f"⚠️  Null values found: {null_counts.sum()}")
    
    # Check data types
    logger.info(f"Data types: {df.dtypes.to_dict()}")
    
    # Check value ranges for sensors
    sensor_columns = [f's{i}' for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
    for sensor in sensor_columns:
        if sensor in df.columns:
            min_val = df[sensor].min()
            max_val = df[sensor].max()
            logger.info(f"{sensor}: {min_val:.2f} - {max_val:.2f}")
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(f"Memory usage: {memory_mb:.2f} MB")
    
    # Check if within free tier limits
    if memory_mb <= 512:
        logger.info("✅ Memory usage within free tier limits (512MB)")
    else:
        logger.warning(f"⚠️  Memory usage exceeds free tier limits: {memory_mb:.2f} MB")

def main():
    """Main test function"""
    try:
        # Setup logging
        logger = setup_logging()
        logger.info("Starting C-MAPSS Data Generation Test...")
        
        # Generate sample data
        success = generate_sample_cmapss_data()
        
        if success:
            logger.info("✅ C-MAPSS data generation test completed successfully!")
            logger.info("Check the generated files in data/raw/ and data/processed/ directories")
        else:
            logger.error("❌ C-MAPSS data generation test failed")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error in test execution: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
