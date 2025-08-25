"""
Performance Testing for Free Tier Constraints
Predictive Maintenance System
"""

import logging
import os
import sys
import time
import psutil
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import get_config
from scripts.cmapss_loader import CMAPSSLoader
from scripts.supabase_connector import SupabaseConnector
from scripts.data_quality_validator import DataQualityValidator

# =====================================================
# Logging Setup
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================
# Performance Tester Class
# =====================================================

class PerformanceTester:
    """Test performance within free tier constraints"""
    
    def __init__(self):
        self.config = get_config()
        self.metrics = {
            'memory_tests': [],
            'throughput_tests': [],
            'latency_tests': [],
            'constraint_violations': []
        }
        
        # Free tier limits
        self.limits = {
            'max_memory_mb': 512,  # Conservative limit for t2.micro (1GB total)
            'max_db_size_mb': 500,  # Supabase free tier
            'max_s3_size_gb': 5,    # AWS S3 free tier
            'max_batch_size': 1000,  # Records per batch
            'max_api_calls_per_second': 10  # Rate limiting
        }
    
    def run_all_tests(self) -> Dict:
        """Run all performance tests"""
        logger.info("=" * 60)
        logger.info("PERFORMANCE TESTING - FREE TIER CONSTRAINTS")
        logger.info("=" * 60)
        
        test_results = {
            'passed': True,
            'tests_run': 0,
            'tests_passed': 0,
            'details': {}
        }
        
        # List of tests to run
        tests = [
            ("Memory Usage Test", self.test_memory_usage),
            ("Data Loading Performance", self.test_data_loading_performance),
            ("Batch Processing Performance", self.test_batch_processing),
            ("Database Ingestion Rate", self.test_database_ingestion),
            ("Data Quality Validation Performance", self.test_validation_performance),
            ("Concurrent Operations", self.test_concurrent_operations)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\nRunning: {test_name}")
            logger.info("-" * 40)
            
            try:
                result = test_func()
                test_results['tests_run'] += 1
                
                if result['passed']:
                    test_results['tests_passed'] += 1
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    test_results['passed'] = False
                    logger.error(f"❌ {test_name} FAILED")
                
                test_results['details'][test_name] = result
                
            except Exception as e:
                logger.error(f"❌ {test_name} ERROR: {e}")
                test_results['passed'] = False
                test_results['details'][test_name] = {
                    'passed': False,
                    'error': str(e)
                }
        
        # Generate report
        self._generate_performance_report(test_results)
        
        return test_results
    
    def test_memory_usage(self) -> Dict:
        """Test memory usage stays within limits"""
        result = {
            'passed': True,
            'max_memory_used_mb': 0,
            'limit_mb': self.limits['max_memory_mb'],
            'details': []
        }
        
        try:
            # Get initial memory
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Test 1: Load small dataset
            loader = CMAPSSLoader(self.config.cmapss)
            loader.config.num_engines = 10  # Small dataset
            
            # Generate data
            data = loader._generate_sample_training_data()
            memory_after_load = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = memory_after_load - initial_memory
            
            result['details'].append({
                'test': 'Small dataset (10 engines)',
                'memory_used_mb': memory_used,
                'passed': memory_used < self.limits['max_memory_mb']
            })
            
            # Test 2: Load medium dataset
            loader.config.num_engines = 50
            data = loader._generate_sample_training_data()
            memory_after_medium = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = memory_after_medium - initial_memory
            
            result['details'].append({
                'test': 'Medium dataset (50 engines)',
                'memory_used_mb': memory_used,
                'passed': memory_used < self.limits['max_memory_mb']
            })
            
            # Test 3: Load full dataset
            loader.config.num_engines = 100
            data = loader._generate_sample_training_data()
            memory_after_full = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = memory_after_full - initial_memory
            
            result['max_memory_used_mb'] = memory_used
            
            result['details'].append({
                'test': 'Full dataset (100 engines)',
                'memory_used_mb': memory_used,
                'passed': memory_used < self.limits['max_memory_mb']
            })
            
            # Check if any test exceeded limits
            for detail in result['details']:
                if not detail['passed']:
                    result['passed'] = False
                    self.metrics['constraint_violations'].append(
                        f"Memory limit exceeded: {detail['memory_used_mb']:.2f}MB > {self.limits['max_memory_mb']}MB"
                    )
            
            # Cleanup
            loader.cleanup_memory()
            
        except Exception as e:
            result['passed'] = False
            result['error'] = str(e)
        
        return result
    
    def test_data_loading_performance(self) -> Dict:
        """Test data loading performance"""
        result = {
            'passed': True,
            'load_times': [],
            'throughput_records_per_second': 0
        }
        
        try:
            loader = CMAPSSLoader(self.config.cmapss)
            
            # Test different batch sizes
            batch_sizes = [100, 500, 1000]
            
            for batch_size in batch_sizes:
                loader.batch_size = batch_size
                
                start_time = time.time()
                data = loader.load_training_data()
                load_time = time.time() - start_time
                
                if data is not None:
                    records_per_second = len(data) / load_time
                    
                    result['load_times'].append({
                        'batch_size': batch_size,
                        'records': len(data),
                        'time_seconds': load_time,
                        'throughput': records_per_second
                    })
                    
                    result['throughput_records_per_second'] = max(
                        result['throughput_records_per_second'],
                        records_per_second
                    )
            
            # Check if performance is acceptable (at least 1000 records/second)
            min_throughput = 1000
            if result['throughput_records_per_second'] < min_throughput:
                result['passed'] = False
                self.metrics['constraint_violations'].append(
                    f"Low throughput: {result['throughput_records_per_second']:.0f} records/s < {min_throughput} records/s"
                )
            
            # Cleanup
            loader.cleanup_memory()
            
        except Exception as e:
            result['passed'] = False
            result['error'] = str(e)
        
        return result
    
    def test_batch_processing(self) -> Dict:
        """Test batch processing performance"""
        result = {
            'passed': True,
            'optimal_batch_size': 0,
            'batch_performance': []
        }
        
        try:
            # Create sample data
            num_records = 10000
            data = pd.DataFrame({
                'unit_number': np.repeat(range(1, 11), num_records // 10),
                'time_cycles': np.tile(range(1, num_records // 10 + 1), 10),
                's2': np.random.normal(640, 10, num_records),
                's3': np.random.normal(1600, 20, num_records)
            })
            
            # Test different batch sizes
            batch_sizes = [100, 250, 500, 1000]
            best_performance = float('inf')
            
            for batch_size in batch_sizes:
                start_time = time.time()
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Process in batches
                for i in range(0, len(data), batch_size):
                    batch = data.iloc[i:i+batch_size]
                    # Simulate processing
                    _ = batch.describe()
                
                process_time = time.time() - start_time
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                memory_used = memory_after - memory_before
                
                performance_metric = process_time + (memory_used / 100)  # Combined metric
                
                result['batch_performance'].append({
                    'batch_size': batch_size,
                    'time_seconds': process_time,
                    'memory_used_mb': memory_used,
                    'performance_score': performance_metric
                })
                
                if performance_metric < best_performance:
                    best_performance = performance_metric
                    result['optimal_batch_size'] = batch_size
            
            # Check if batch processing is within limits
            max_batch_memory = max(p['memory_used_mb'] for p in result['batch_performance'])
            if max_batch_memory > self.limits['max_memory_mb']:
                result['passed'] = False
                self.metrics['constraint_violations'].append(
                    f"Batch processing exceeded memory limit: {max_batch_memory:.2f}MB"
                )
            
        except Exception as e:
            result['passed'] = False
            result['error'] = str(e)
        
        return result
    
    def test_database_ingestion(self) -> Dict:
        """Test database ingestion rate"""
        result = {
            'passed': True,
            'ingestion_rate': 0,
            'test_skipped': False
        }
        
        try:
            # Check if Supabase is configured
            if not self.config.database.url or not self.config.database.key:
                result['test_skipped'] = True
                result['message'] = "Supabase not configured, skipping test"
                logger.warning("⚠️  Supabase not configured, skipping database ingestion test")
                return result
            
            connector = SupabaseConnector(self.config.database)
            
            # Test connection first
            if not connector.test_connection():
                result['test_skipped'] = True
                result['message'] = "Database connection failed"
                return result
            
            # Create small test dataset
            test_data = pd.DataFrame({
                'unit_number': [1, 1, 1, 2, 2, 2],
                'time_cycles': [1, 2, 3, 1, 2, 3],
                'timestamp': pd.date_range('2024-01-01', periods=6, freq='H'),
                'altitude': [35000] * 6,
                's2': [640, 641, 642, 639, 640, 641],
                's3': [1600, 1601, 1602, 1599, 1600, 1601],
                'rul': [100, 99, 98, 100, 99, 98]
            })
            
            # Measure ingestion time
            start_time = time.time()
            success, results = connector.insert_sensor_data_batch(test_data)
            ingestion_time = time.time() - start_time
            
            if success:
                result['ingestion_rate'] = results['records_inserted'] / ingestion_time
                
                # Check if rate is acceptable (at least 100 records/second)
                min_rate = 100
                if result['ingestion_rate'] < min_rate:
                    result['passed'] = False
                    self.metrics['constraint_violations'].append(
                        f"Low ingestion rate: {result['ingestion_rate']:.0f} records/s < {min_rate} records/s"
                    )
            else:
                result['passed'] = False
                result['error'] = "Ingestion failed"
            
            # Cleanup
            connector.close()
            
        except Exception as e:
            result['passed'] = False
            result['error'] = str(e)
        
        return result
    
    def test_validation_performance(self) -> Dict:
        """Test data quality validation performance"""
        result = {
            'passed': True,
            'validation_times': [],
            'average_time_seconds': 0
        }
        
        try:
            validator = DataQualityValidator()
            
            # Test with different dataset sizes
            sizes = [100, 500, 1000, 5000]
            
            for size in sizes:
                # Create test data
                test_data = pd.DataFrame({
                    'unit_number': np.repeat(range(1, size // 10 + 1), 10),
                    'time_cycles': np.tile(range(1, 11), size // 10),
                    's2': np.random.normal(640, 10, size),
                    's3': np.random.normal(1600, 20, size)
                })
                
                # Measure validation time
                start_time = time.time()
                validation_result = validator.validate_dataframe(test_data)
                validation_time = time.time() - start_time
                
                result['validation_times'].append({
                    'size': size,
                    'time_seconds': validation_time,
                    'quality_score': validation_result.get('quality_score', 0)
                })
            
            # Calculate average
            result['average_time_seconds'] = np.mean([
                t['time_seconds'] for t in result['validation_times']
            ])
            
            # Check if validation is fast enough (< 1 second for 1000 records)
            for timing in result['validation_times']:
                if timing['size'] == 1000 and timing['time_seconds'] > 1.0:
                    result['passed'] = False
                    self.metrics['constraint_violations'].append(
                        f"Slow validation: {timing['time_seconds']:.2f}s for 1000 records"
                    )
            
        except Exception as e:
            result['passed'] = False
            result['error'] = str(e)
        
        return result
    
    def test_concurrent_operations(self) -> Dict:
        """Test concurrent operations performance"""
        result = {
            'passed': True,
            'concurrent_memory_mb': 0,
            'operations_completed': 0
        }
        
        try:
            import concurrent.futures
            
            def simulate_operation(operation_id):
                """Simulate a data processing operation"""
                # Create small dataset
                data = pd.DataFrame({
                    'id': range(100),
                    'value': np.random.randn(100)
                })
                # Process data
                result = data.describe()
                return operation_id, len(data)
            
            # Measure memory before
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Run concurrent operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(simulate_operation, i) for i in range(5)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            # Measure memory after
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before
            
            result['concurrent_memory_mb'] = memory_used
            result['operations_completed'] = len(results)
            
            # Check if memory usage is within limits
            if memory_used > self.limits['max_memory_mb']:
                result['passed'] = False
                self.metrics['constraint_violations'].append(
                    f"Concurrent operations exceeded memory limit: {memory_used:.2f}MB"
                )
            
        except Exception as e:
            result['passed'] = False
            result['error'] = str(e)
        
        return result
    
    def _generate_performance_report(self, test_results: Dict):
        """Generate performance testing report"""
        try:
            report = []
            report.append("=" * 60)
            report.append("PERFORMANCE TESTING REPORT")
            report.append("=" * 60)
            report.append(f"Generated: {datetime.utcnow().isoformat()}")
            report.append("")
            
            # Summary
            report.append("SUMMARY:")
            report.append(f"  Status: {'✅ PASSED' if test_results['passed'] else '❌ FAILED'}")
            report.append(f"  Tests Run: {test_results['tests_run']}")
            report.append(f"  Tests Passed: {test_results['tests_passed']}")
            report.append("")
            
            # Free Tier Limits
            report.append("FREE TIER LIMITS:")
            for limit_name, limit_value in self.limits.items():
                report.append(f"  - {limit_name}: {limit_value}")
            report.append("")
            
            # Test Results
            report.append("TEST RESULTS:")
            for test_name, result in test_results['details'].items():
                report.append(f"\n{test_name}:")
                report.append(f"  Status: {'✅ PASSED' if result.get('passed', False) else '❌ FAILED'}")
                
                if 'error' in result:
                    report.append(f"  Error: {result['error']}")
                
                # Add specific test details
                if 'max_memory_used_mb' in result:
                    report.append(f"  Max Memory Used: {result['max_memory_used_mb']:.2f} MB")
                
                if 'throughput_records_per_second' in result:
                    report.append(f"  Throughput: {result['throughput_records_per_second']:.0f} records/s")
                
                if 'optimal_batch_size' in result:
                    report.append(f"  Optimal Batch Size: {result['optimal_batch_size']}")
                
                if 'ingestion_rate' in result:
                    report.append(f"  Ingestion Rate: {result['ingestion_rate']:.0f} records/s")
            
            # Constraint Violations
            if self.metrics['constraint_violations']:
                report.append("\nCONSTRAINT VIOLATIONS:")
                for violation in self.metrics['constraint_violations']:
                    report.append(f"  ⚠️  {violation}")
            
            # Recommendations
            report.append("\nRECOMMENDATIONS:")
            if test_results['passed']:
                report.append("  ✅ System is operating within free tier constraints")
                report.append("  - Current configuration is suitable for production")
            else:
                report.append("  ❌ System exceeded free tier constraints")
                report.append("  - Consider reducing batch sizes")
                report.append("  - Implement more aggressive memory management")
                report.append("  - Monitor resource usage closely in production")
            
            report.append("")
            report.append("=" * 60)
            
            # Save report
            report_text = "\n".join(report)
            report_file = f"logs/performance_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
            
            os.makedirs("logs", exist_ok=True)
            with open(report_file, 'w') as f:
                f.write(report_text)
            
            logger.info(f"Performance report saved to: {report_file}")
            print("\n" + report_text)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")

# =====================================================
# Main Function
# =====================================================

def main():
    """Main function for performance testing"""
    logger.info("Starting Performance Testing...")
    
    tester = PerformanceTester()
    results = tester.run_all_tests()
    
    if results['passed']:
        logger.info("✅ All performance tests passed!")
        return True
    else:
        logger.error(f"❌ {results['tests_run'] - results['tests_passed']} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)