"""
C-MAPSS Dataset Loader - Predictive Maintenance System
Free Tier Optimized for t2.micro EC2 instances
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from config.config import CMAPSSConfig, get_config

# =====================================================
# Logging Setup
# =====================================================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/cmapss_loader.log')
        ]
    )
    return logging.getLogger(__name__)

# =====================================================
# C-MAPSS Data Loader Class
# =====================================================

class CMAPSSLoader:
    """C-MAPSS dataset loader with free tier optimizations"""
    
    def __init__(self, config: CMAPSSConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Free tier optimizations
        self.batch_size = 500  # Optimized for t2.micro memory
        self.max_memory_mb = 512  # Stay within 1GB limit
        self.stats = {
            'total_records': 0,
            'processing_time_ms': 0,
            'memory_usage_mb': 0.0,
            'batches_processed': 0
        }
        
        # Ensure output directories exist
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    
    def download_cmapss_data(self, output_dir: str = "data/raw") -> bool:
        """Download or create sample C-MAPSS data"""
        try:
            self.logger.info("Starting C-MAPSS data download/creation...")
            
            # Check if data already exists
            if self._check_existing_data(output_dir):
                self.logger.info("C-MAPSS data already exists, skipping download")
                return True
            
            # Create sample data for free tier testing
            success = self._create_sample_cmapss_data(output_dir)
            
            if success:
                self.logger.info("✅ C-MAPSS data created successfully")
                return True
            else:
                self.logger.error("❌ Failed to create C-MAPSS data")
                return False
                
        except Exception as e:
            self.logger.error(f"Error creating sample data: {e}")
            return False
    
    def _check_existing_data(self, output_dir: str) -> bool:
        """Check if C-MAPSS data files already exist"""
        required_files = [
            os.path.join(output_dir, self.config.train_file),
            os.path.join(output_dir, self.config.test_file),
            os.path.join(output_dir, self.config.rul_file)
        ]
        
        return all(os.path.exists(f) for f in required_files)
    
    def _create_sample_cmapss_data(self, output_dir: str) -> bool:
        """Create sample C-MAPSS data for free tier testing"""
        try:
            self.logger.info("Creating sample C-MAPSS data...")
            
            # Generate training data
            train_data = self._generate_sample_training_data()
            if train_data is None:
                return False
            
            # Generate test data
            test_data = self._generate_sample_test_data()
            if test_data is None:
                return False
            
            # Generate RUL data
            rul_data = self._generate_sample_rul_data()
            if rul_data is None:
                return False
            
            # Save data files
            self._save_data_files(train_data, test_data, rul_data, output_dir)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating sample data: {e}")
            return False
    
    def _generate_sample_training_data(self) -> pd.DataFrame:
        """Generate sample training data based on C-MAPSS characteristics"""
        try:
            data = []
            
            # Generate data for each engine
            for engine_num in range(1, self.config.num_engines + 1):
                # Random number of cycles for this engine (50-500)
                max_cycles = np.random.randint(50, self.config.max_cycles + 1)
                
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
            df = pd.DataFrame(data, columns=self.config.column_names)
            
            # Add RUL and timestamps
            df = self._add_rul_labels(df)
            df = self._add_timestamps(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating training data: {e}")
            return None
    
    def _generate_sample_test_data(self) -> pd.DataFrame:
        """Generate sample test data"""
        try:
            # Similar to training data but with fewer engines
            test_engines = min(20, self.config.num_engines // 5)
            
            data = []
            for engine_num in range(1, test_engines + 1):
                max_cycles = np.random.randint(30, 100)
                
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
            
            df = pd.DataFrame(data, columns=self.config.column_names)
            df = self._add_rul_labels(df)
            df = self._add_timestamps(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating test data: {e}")
            return None
    
    def _generate_sample_rul_data(self) -> pd.DataFrame:
        """Generate sample RUL data"""
        try:
            data = []
            
            for engine_num in range(1, self.config.num_engines + 1):
                # Random RUL value (0-125 cycles)
                rul = np.random.randint(0, 126)
                data.append([engine_num, rul])
            
            df = pd.DataFrame(data, columns=['unit_number', 'rul'])
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating RUL data: {e}")
            return None
    
    def _add_rul_labels(self, df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
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
            self.logger.error(f"Error adding RUL labels: {e}")
            return df
    
    def _add_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic timestamps based on operational cycles"""
        try:
            # Assume each cycle represents 1 day of operation
            from datetime import datetime, timedelta
            base_date = datetime(2020, 1, 1)
            
            def calculate_timestamp(row):
                return base_date + timedelta(days=int(row['time_cycles']))
            
            df['timestamp'] = df.apply(calculate_timestamp, axis=1)
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding timestamps: {e}")
            return df
    
    def _save_data_files(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                         rul_data: pd.DataFrame, output_dir: str):
        """Save generated data to files"""
        try:
            # Save training data
            train_file = os.path.join(output_dir, self.config.train_file)
            train_data.to_csv(train_file, sep=' ', header=False, index=False)
            self.logger.info(f"Training data saved to {train_file}")
            
            # Save test data
            test_file = os.path.join(output_dir, self.config.test_file)
            test_data.to_csv(test_file, sep=' ', header=False, index=False)
            self.logger.info(f"Test data saved to {test_file}")
            
            # Save RUL data
            rul_file = os.path.join(output_dir, self.config.rul_file)
            rul_data.to_csv(rul_file, sep=' ', header=False, index=False)
            self.logger.info(f"RUL data saved to {rul_file}")
            
            # Save processed data as parquet
            output_file = os.path.join("data/processed", "training_data_processed.parquet")
            train_data.to_parquet(output_file, compression='snappy')
            self.logger.info(f"Processed data saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving data files: {e}")
            raise
    
    def load_training_data(self, file_path: str = None) -> pd.DataFrame:
        """Load training data with memory optimization"""
        try:
            if file_path is None:
                file_path = os.path.join("data/raw", self.config.train_file)
            
            self.logger.info(f"Loading training data from {file_path}")
            
            # Load data in chunks for memory efficiency
            chunks = []
            chunk_size = self.batch_size
            
            for chunk in pd.read_csv(file_path, sep=' ', header=None, 
                                   names=self.config.column_names, 
                                   chunksize=chunk_size):
                chunks.append(chunk)
                
                # Check memory usage
                if self._check_memory_limit():
                    self.logger.warning("Memory limit approaching, processing chunks...")
                    break
            
            # Combine chunks
            df = pd.concat(chunks, ignore_index=True)
            
            # Update stats
            self.stats['total_records'] = len(df)
            self.stats['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            self.logger.info(f"Loaded {len(df)} training records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")
            return None
    
    def validate_data(self, df: pd.DataFrame) -> dict:
        """Validate data quality"""
        try:
            self.logger.info("Validating data quality...")
            
            validation_results = {
                'total_records': len(df),
                'null_counts': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.to_dict(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'quality_score': 1.0,
                'issues': []
            }
            
            # Check for null values
            null_count = df.isnull().sum().sum()
            if null_count > 0:
                validation_results['issues'].append(f"Found {null_count} null values")
                validation_results['quality_score'] -= 0.1
            
            # Check data types
            expected_dtypes = self.config.dtypes
            for col, expected_dtype in expected_dtypes.items():
                if col in df.columns:
                    actual_dtype = str(df[col].dtype)
                    if expected_dtype not in actual_dtype:
                        validation_results['issues'].append(
                            f"Column {col}: expected {expected_dtype}, got {actual_dtype}"
                        )
                        validation_results['quality_score'] -= 0.05
            
            # Check memory usage
            if validation_results['memory_usage_mb'] > self.max_memory_mb:
                validation_results['issues'].append(
                    f"Memory usage {validation_results['memory_usage_mb']:.2f}MB exceeds limit {self.max_memory_mb}MB"
                )
                validation_results['quality_score'] -= 0.2
            
            # Ensure quality score is not negative
            validation_results['quality_score'] = max(0.0, validation_results['quality_score'])
            
            self.logger.info(f"Data validation completed. Quality score: {validation_results['quality_score']:.2f}")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating data: {e}")
            return {'quality_score': 0.0, 'issues': [str(e)]}
    
    def _check_memory_limit(self) -> bool:
        """Check if memory usage is approaching the limit"""
        import psutil
        memory = psutil.virtual_memory()
        return memory.percent > 80  # Warning at 80% usage
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage statistics"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            return {
                'total_memory_mb': memory.total / 1024 / 1024,
                'available_memory_mb': memory.available / 1024 / 1024,
                'used_memory_mb': memory.used / 1024 / 1024,
                'memory_percent': memory.percent,
                'free_tier_limit_mb': 1024,  # t2.micro has 1GB
                'recommended_limit_mb': self.max_memory_mb
            }
        except ImportError:
            return {'total_memory_mb': 0.0, 'free_tier_limit_mb': 1024}
    
    def cleanup_memory(self):
        """Clean up memory and reset stats"""
        import gc
        gc.collect()
        
        self.stats = {
            'total_records': 0,
            'processing_time_ms': 0,
            'memory_usage_mb': 0.0,
            'batches_processed': 0
        }
        
        self.logger.info("Memory cleanup completed")

# =====================================================
# Main Function
# =====================================================

def main():
    """Main function for C-MAPSS data loading"""
    try:
        # Setup logging
        logger = setup_logging()
        logger.info("Starting C-MAPSS Data Loader...")
        
        # Get configuration
        config = get_config()
        
        # Create loader
        loader = CMAPSSLoader(config.cmapss)
        
        # Download/create data
        if not loader.download_cmapss_data():
            logger.error("Failed to download/create C-MAPSS data")
            return False
        
        # Load training data
        train_data = loader.load_training_data()
        if train_data is None:
            logger.error("Failed to load training data")
            return False
        
        # Validate data
        validation_results = loader.validate_data(train_data)
        logger.info(f"Data validation results: {validation_results}")
        
        # Cleanup
        loader.cleanup_memory()
        
        logger.info("✅ C-MAPSS data loading completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
