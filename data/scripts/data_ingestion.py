"""
Data Ingestion Pipeline - Predictive Maintenance System
Loads C-MAPSS data into Supabase with data quality validation
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import get_config
from scripts.cmapss_loader import CMAPSSLoader
from scripts.supabase_connector import SupabaseConnector

# =====================================================
# Logging Setup
# =====================================================

# Create logs directory if it doesn't exist
import os
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/data_ingestion.log')
    ]
)
logger = logging.getLogger(__name__)

# =====================================================
# Data Ingestion Pipeline Class
# =====================================================

class DataIngestionPipeline:
    """Main data ingestion pipeline for C-MAPSS to Supabase"""
    
    def __init__(self):
        self.config = get_config()
        self.cmapss_loader = CMAPSSLoader(self.config.cmapss)
        self.db_connector = SupabaseConnector(self.config.database)
        
        # Set base directory (data/ directory)
        self.base_dir = Path(__file__).parent.parent
        
        # Pipeline metrics
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'total_records_loaded': 0,
            'total_records_inserted': 0,
            'data_quality_score': 0.0,
            'errors': [],
            'warnings': []
        }
        
        # Ensure directories exist
        os.makedirs(self.base_dir / 'logs', exist_ok=True)
        os.makedirs(self.base_dir / 'raw', exist_ok=True)
        os.makedirs(self.base_dir / 'processed', exist_ok=True)
    
    def run_full_pipeline(self) -> Tuple[bool, Dict]:
        """Run the complete data ingestion pipeline"""
        self.metrics['start_time'] = datetime.utcnow()
        logger.info("=" * 60)
        logger.info("Starting Data Ingestion Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Test database connection
            if not self._test_connections():
                return False, self.metrics
            
            # Step 2: Download/Create C-MAPSS data
            if not self._prepare_cmapss_data():
                return False, self.metrics
            
            # Step 3: Load and validate data
            data = self._load_and_validate_data()
            if data is None or data.empty:
                return False, self.metrics
            
            # Step 4: Perform data quality checks
            if not self._perform_data_quality_checks(data):
                logger.warning("Data quality checks failed, but continuing...")
            
            # Step 5: Ingest data into Supabase
            if not self._ingest_to_supabase(data):
                return False, self.metrics
            
            # Step 6: Verify ingestion
            if not self._verify_ingestion():
                return False, self.metrics
            
            # Step 7: Log metrics
            self._log_pipeline_metrics()
            
            self.metrics['end_time'] = datetime.utcnow()
            duration = (self.metrics['end_time'] - self.metrics['start_time']).total_seconds()
            
            logger.info("=" * 60)
            logger.info(f"✅ Pipeline completed successfully in {duration:.2f} seconds")
            logger.info(f"Total records inserted: {self.metrics['total_records_inserted']}")
            logger.info(f"Data quality score: {self.metrics['data_quality_score']:.2f}")
            logger.info("=" * 60)
            
            return True, self.metrics
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.metrics['errors'].append(str(e))
            return False, self.metrics
        
        finally:
            # Cleanup
            self._cleanup()
    
    def _test_connections(self) -> bool:
        """Test database connections"""
        logger.info("Testing database connections...")
        
        try:
            # Test Supabase connection
            if not self.db_connector.test_connection():
                logger.error("❌ Database connection test failed")
                self.metrics['errors'].append("Database connection failed")
                return False
            
            logger.info("✅ Database connections successful")
            return True
            
        except Exception as e:
            logger.error(f"Connection test error: {e}")
            self.metrics['errors'].append(f"Connection test: {e}")
            return False
    
    def _prepare_cmapss_data(self) -> bool:
        """Download or create C-MAPSS data"""
        logger.info("Preparing C-MAPSS data...")
        
        try:
            # Download/create data
            if not self.cmapss_loader.download_cmapss_data():
                logger.error("❌ Failed to prepare C-MAPSS data")
                self.metrics['errors'].append("C-MAPSS data preparation failed")
                return False
            
            logger.info("✅ C-MAPSS data ready")
            return True
            
        except Exception as e:
            logger.error(f"Data preparation error: {e}")
            self.metrics['errors'].append(f"Data preparation: {e}")
            return False
    
    def _load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate C-MAPSS data"""
        logger.info("Loading and validating data...")
        
        try:
            # Load training data
            data = self.cmapss_loader.load_training_data()
            
            if data is None or data.empty:
                logger.error("❌ No data loaded")
                self.metrics['errors'].append("No data loaded")
                return None
            
            self.metrics['total_records_loaded'] = len(data)
            
            # Validate data
            validation_results = self.cmapss_loader.validate_data(data)
            self.metrics['data_quality_score'] = validation_results.get('quality_score', 0.0)
            
            if validation_results['quality_score'] < 0.5:
                logger.error(f"❌ Data quality too low: {validation_results['quality_score']}")
                self.metrics['errors'].append(f"Low data quality: {validation_results}")
                return None
            
            # Add any issues as warnings
            for issue in validation_results.get('issues', []):
                self.metrics['warnings'].append(issue)
            
            logger.info(f"✅ Loaded {len(data)} records with quality score: {validation_results['quality_score']:.2f}")
            return data
            
        except Exception as e:
            logger.error(f"Data loading error: {e}")
            self.metrics['errors'].append(f"Data loading: {e}")
            return None
    
    def _perform_data_quality_checks(self, data: pd.DataFrame) -> bool:
        """Perform additional data quality checks"""
        logger.info("Performing data quality checks...")
        
        try:
            quality_metrics = {
                'table_name': 'sensor_data',
                'total_records': len(data),
                'null_count': int(data.isnull().sum().sum()),
                'duplicate_count': int(data.duplicated().sum()),
                'anomaly_count': 0,
                'quality_score': self.metrics['data_quality_score'],
                'check_results': {},
                'issues_found': []
            }
            
            # Check for nulls in critical columns
            critical_columns = ['unit_number', 'time_cycles']
            for col in critical_columns:
                if col in data.columns:
                    null_count = data[col].isnull().sum()
                    if null_count > 0:
                        issue = f"Found {null_count} nulls in critical column: {col}"
                        quality_metrics['issues_found'].append(issue)
                        logger.warning(issue)
            
            # Check for duplicates
            dup_cols = ['unit_number', 'time_cycles']
            if all(col in data.columns for col in dup_cols):
                duplicates = data.duplicated(subset=dup_cols, keep=False)
                if duplicates.any():
                    dup_count = duplicates.sum()
                    quality_metrics['duplicate_count'] = int(dup_count)
                    issue = f"Found {dup_count} duplicate records"
                    quality_metrics['issues_found'].append(issue)
                    logger.warning(issue)
            
            # Check sensor value ranges
            sensor_ranges = self.config.data_processing.sensor_ranges
            for sensor, (min_val, max_val) in sensor_ranges.items():
                if sensor in data.columns:
                    out_of_range = ((data[sensor] < min_val) | (data[sensor] > max_val)).sum()
                    if out_of_range > 0:
                        pct = (out_of_range / len(data)) * 100
                        if pct > 5:  # More than 5% out of range
                            issue = f"Sensor {sensor}: {out_of_range} values ({pct:.1f}%) out of range"
                            quality_metrics['issues_found'].append(issue)
                            quality_metrics['anomaly_count'] += out_of_range
                            logger.warning(issue)
            
            # Log quality metrics to database
            self.db_connector.log_data_quality_metrics(quality_metrics)
            
            # Determine if quality is acceptable
            major_issues = len([i for i in quality_metrics['issues_found'] if 'critical' in i.lower()])
            
            if major_issues > 0:
                logger.error(f"❌ Found {major_issues} major data quality issues")
                return False
            
            logger.info(f"✅ Data quality checks completed with {len(quality_metrics['issues_found'])} minor issues")
            return True
            
        except Exception as e:
            logger.error(f"Data quality check error: {e}")
            self.metrics['errors'].append(f"Quality check: {e}")
            return False
    
    def _ingest_to_supabase(self, data: pd.DataFrame) -> bool:
        """Ingest data into Supabase database"""
        logger.info("Starting data ingestion to Supabase...")
        
        try:
            # Process data in batches for free tier optimization
            batch_size = self.config.data_processing.batch_size
            total_batches = (len(data) + batch_size - 1) // batch_size
            
            logger.info(f"Processing {len(data)} records in {total_batches} batches of {batch_size}")
            
            total_inserted = 0
            failed_batches = 0
            
            for i in range(0, len(data), batch_size):
                batch_num = (i // batch_size) + 1
                batch_data = data.iloc[i:i+batch_size].copy()
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_data)} records)")
                
                # Insert batch
                success, results = self.db_connector.insert_sensor_data_batch(batch_data)
                
                if success:
                    total_inserted += results.get('records_inserted', 0)
                    logger.info(f"✅ Batch {batch_num} inserted: {results['records_inserted']} records")
                else:
                    failed_batches += 1
                    logger.error(f"❌ Batch {batch_num} failed: {results.get('errors', [])}")
                    self.metrics['warnings'].extend(results.get('errors', []))
                
                # Add small delay to avoid rate limiting
                time.sleep(0.5)
            
            self.metrics['total_records_inserted'] = total_inserted
            
            if total_inserted == 0:
                logger.error("❌ No records were inserted")
                self.metrics['errors'].append("No records inserted")
                return False
            
            success_rate = (total_inserted / len(data)) * 100
            logger.info(f"✅ Ingestion completed: {total_inserted}/{len(data)} records ({success_rate:.1f}% success rate)")
            
            if failed_batches > 0:
                logger.warning(f"⚠️  {failed_batches} batches failed during ingestion")
            
            return success_rate >= 50  # Accept if at least 50% successful
            
        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            self.metrics['errors'].append(f"Ingestion: {e}")
            return False
    
    def _verify_ingestion(self) -> bool:
        """Verify data was successfully ingested"""
        logger.info("Verifying data ingestion...")
        
        try:
            # Check a sample engine
            sample_data = self.db_connector.get_latest_sensor_data(unit_number=1, limit=10)
            
            if sample_data.empty:
                logger.warning("⚠️  No data found for verification")
                self.metrics['warnings'].append("Could not verify ingestion")
                return True  # Don't fail if we can't verify
            
            logger.info(f"✅ Verification successful: Found {len(sample_data)} sample records")
            return True
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            self.metrics['warnings'].append(f"Verification: {e}")
            return True  # Don't fail on verification
    
    def _log_pipeline_metrics(self):
        """Log pipeline execution metrics"""
        try:
            metrics = {
                'table_name': 'pipeline_execution',
                'total_records': self.metrics['total_records_inserted'],
                'quality_score': self.metrics['data_quality_score'],
                'check_results': {
                    'records_loaded': self.metrics['total_records_loaded'],
                    'records_inserted': self.metrics['total_records_inserted'],
                    'errors': len(self.metrics['errors']),
                    'warnings': len(self.metrics['warnings'])
                },
                'issues_found': self.metrics['errors'][:10]  # Log first 10 errors
            }
            
            self.db_connector.log_data_quality_metrics(metrics)
            logger.info("Pipeline metrics logged to database")
            
        except Exception as e:
            logger.warning(f"Could not log pipeline metrics: {e}")
    
    def _cleanup(self):
        """Cleanup resources"""
        try:
            # Cleanup loader
            self.cmapss_loader.cleanup_memory()
            
            # Close database connections
            self.db_connector.close()
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
    
    def run_incremental_update(self, start_date: str = None) -> Tuple[bool, Dict]:
        """Run incremental data update (for new data only)"""
        logger.info("Running incremental data update...")
        
        # This would be implemented for real-time data updates
        # For now, it's a placeholder for future implementation
        
        self.metrics['warnings'].append("Incremental updates not yet implemented")
        return True, self.metrics

# =====================================================
# Main Function
# =====================================================

def main():
    """Main function for data ingestion pipeline"""
    try:
        # Create pipeline
        pipeline = DataIngestionPipeline()
        
        # Run full pipeline
        success, metrics = pipeline.run_full_pipeline()
        
        # Print summary
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"Records Loaded: {metrics['total_records_loaded']}")
        print(f"Records Inserted: {metrics['total_records_inserted']}")
        print(f"Data Quality Score: {metrics['data_quality_score']:.2f}")
        print(f"Errors: {len(metrics['errors'])}")
        print(f"Warnings: {len(metrics['warnings'])}")
        
        if metrics['errors']:
            print("\nErrors:")
            for error in metrics['errors'][:5]:
                print(f"  - {error}")
        
        if metrics['warnings']:
            print("\nWarnings:")
            for warning in metrics['warnings'][:5]:
                print(f"  - {warning}")
        
        print("=" * 60)
        
        return success
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)