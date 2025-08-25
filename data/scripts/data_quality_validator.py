"""
Data Quality Validation with Great Expectations
Predictive Maintenance System - Free Tier Optimized
"""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Try importing Great Expectations with fallback
try:
    import great_expectations as ge
    from great_expectations.core import ExpectationSuite
    from great_expectations.core.expectation_configuration import ExpectationConfiguration
    GE_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Great Expectations not available, using pandas-only validation")
    GE_AVAILABLE = False

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import get_config

# =====================================================
# Logging Setup
# =====================================================

logger = logging.getLogger(__name__)

# =====================================================
# Data Quality Validator Class
# =====================================================

class DataQualityValidator:
    """Data quality validation using Great Expectations"""
    
    def __init__(self):
        self.config = get_config()
        self.context = None
        self.suite_name = "cmapss_sensor_data_suite"
        
        # Validation thresholds
        self.thresholds = {
            'min_quality_score': self.config.data_processing.min_quality_score,
            'max_null_percentage': self.config.data_processing.max_null_percentage,
            'sensor_ranges': self.config.data_processing.sensor_ranges
        }
        
        # Initialize Great Expectations
        self._initialize_ge_context()
    
    def _initialize_ge_context(self):
        """Initialize Great Expectations context"""
        if not GE_AVAILABLE:
            logger.warning("Great Expectations not available, using pandas-only validation")
            self.context = None
            return
            
        try:
            # Create a simple in-memory context (no persistence for free tier)
            self.context = ge.get_context()
            logger.info("Great Expectations context initialized")
            
        except Exception as e:
            logger.warning(f"Could not initialize GE context, using pandas validation: {e}")
            self.context = None
    
    def create_expectation_suite(self):
        """Create expectation suite for C-MAPSS sensor data"""
        if not GE_AVAILABLE:
            return None
            
        try:
            suite = ExpectationSuite(expectation_suite_name=self.suite_name)
            
            # =====================================================
            # Schema Expectations
            # =====================================================
            
            # Expected columns
            expected_columns = [
                'unit_number', 'time_cycles', 'altitude', 'mach_number', 'tra'
            ] + [f's{i}' for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
            
            suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_table_columns_to_match_set",
                    kwargs={
                        "column_set": expected_columns,
                        "exact_match": False  # Allow additional columns
                    }
                )
            )
            
            # =====================================================
            # Data Type Expectations
            # =====================================================
            
            # Integer columns
            for col in ['unit_number', 'time_cycles']:
                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_be_of_type",
                        kwargs={
                            "column": col,
                            "type_": "int"
                        }
                    )
                )
            
            # Float columns
            float_columns = ['altitude', 'mach_number', 'tra'] + \
                          [f's{i}' for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
            
            for col in float_columns:
                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_be_of_type",
                        kwargs={
                            "column": col,
                            "type_": "float"
                        }
                    )
                )
            
            # =====================================================
            # Null Value Expectations
            # =====================================================
            
            # Critical columns should not have nulls
            critical_columns = ['unit_number', 'time_cycles']
            for col in critical_columns:
                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_not_be_null",
                        kwargs={"column": col}
                    )
                )
            
            # Other columns can have limited nulls
            for col in float_columns:
                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_not_be_null",
                        kwargs={
                            "column": col,
                            "mostly": 0.95  # Allow up to 5% nulls
                        }
                    )
                )
            
            # =====================================================
            # Value Range Expectations
            # =====================================================
            
            # Unit number should be positive
            suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": "unit_number",
                        "min_value": 1,
                        "max_value": 1000  # Reasonable upper limit
                    }
                )
            )
            
            # Time cycles should be positive
            suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": "time_cycles",
                        "min_value": 1,
                        "max_value": 10000  # Reasonable upper limit
                    }
                )
            )
            
            # Operational settings ranges
            suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": "altitude",
                        "min_value": 0,
                        "max_value": 50000
                    }
                )
            )
            
            suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": "mach_number",
                        "min_value": 0,
                        "max_value": 1.5
                    }
                )
            )
            
            # Sensor value ranges
            for sensor, (min_val, max_val) in self.thresholds['sensor_ranges'].items():
                # Add some tolerance (20%) for outliers
                tolerance = 0.2
                range_diff = max_val - min_val
                
                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_be_between",
                        kwargs={
                            "column": sensor,
                            "min_value": min_val - (range_diff * tolerance),
                            "max_value": max_val + (range_diff * tolerance),
                            "mostly": 0.95  # Allow 5% outliers
                        }
                    )
                )
            
            # =====================================================
            # Statistical Expectations
            # =====================================================
            
            # Check for reasonable standard deviations
            for sensor in ['s2', 's3', 's4']:
                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_stdev_to_be_between",
                        kwargs={
                            "column": sensor,
                            "min_value": 0.1,  # Should have some variation
                            "max_value": 1000  # But not too much
                        }
                    )
                )
            
            # =====================================================
            # Uniqueness Expectations
            # =====================================================
            
            # Combination of unit_number and time_cycles should be unique
            suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_compound_columns_to_be_unique",
                    kwargs={
                        "column_list": ["unit_number", "time_cycles"]
                    }
                )
            )
            
            logger.info(f"Created expectation suite with {len(suite.expectations)} expectations")
            return suite
            
        except Exception as e:
            logger.error(f"Error creating expectation suite: {e}")
            if GE_AVAILABLE:
                return ExpectationSuite(expectation_suite_name=self.suite_name)
            return None
    
    def validate_dataframe(self, df: pd.DataFrame, suite_name: str = None) -> Dict:
        """Validate a pandas DataFrame using Great Expectations"""
        try:
            if self.context is None:
                # Fallback to pandas validation
                return self._validate_with_pandas(df)
            
            # Use provided suite or default
            if suite_name is None:
                suite_name = self.suite_name
            
            # Create or get expectation suite
            try:
                suite = self.context.get_expectation_suite(suite_name)
            except:
                suite = self.create_expectation_suite()
                self.context.add_expectation_suite(suite)
            
            # Create batch request
            batch_request = RuntimeBatchRequest(
                datasource_name="pandas_datasource",
                data_connector_name="runtime_data_connector",
                data_asset_name="sensor_data",
                runtime_parameters={"batch_data": df},
                batch_identifiers={"default_identifier_name": "default_identifier"}
            )
            
            # Run validation
            validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite=suite
            )
            
            results = validator.validate()
            
            # Parse results
            validation_summary = self._parse_validation_results(results)
            
            logger.info(f"Validation completed: success={validation_summary['success']}, "
                       f"quality_score={validation_summary['quality_score']:.2f}")
            
            return validation_summary
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            # Fallback to pandas validation
            return self._validate_with_pandas(df)
    
    def _validate_with_pandas(self, df: pd.DataFrame) -> Dict:
        """Fallback validation using pandas (when GE is not available)"""
        logger.info("Using pandas validation (Great Expectations fallback)")
        
        validation_results = {
            'success': True,
            'quality_score': 1.0,
            'total_expectations': 0,
            'successful_expectations': 0,
            'failed_expectations': 0,
            'issues': [],
            'statistics': {}
        }
        
        try:
            # Check for required columns
            expected_columns = ['unit_number', 'time_cycles']
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                validation_results['issues'].append(f"Missing columns: {missing_columns}")
                validation_results['quality_score'] -= 0.2
                validation_results['success'] = False
            
            # Check for nulls
            null_counts = df.isnull().sum()
            total_nulls = null_counts.sum()
            if total_nulls > 0:
                null_percentage = (total_nulls / (len(df) * len(df.columns))) * 100
                if null_percentage > self.thresholds['max_null_percentage'] * 100:
                    validation_results['issues'].append(
                        f"High null percentage: {null_percentage:.2f}%"
                    )
                    validation_results['quality_score'] -= 0.1
            
            # Check for duplicates
            if 'unit_number' in df.columns and 'time_cycles' in df.columns:
                duplicates = df.duplicated(subset=['unit_number', 'time_cycles'])
                if duplicates.any():
                    dup_count = duplicates.sum()
                    validation_results['issues'].append(f"Found {dup_count} duplicate records")
                    validation_results['quality_score'] -= 0.1
            
            # Check sensor ranges
            for sensor, (min_val, max_val) in self.thresholds['sensor_ranges'].items():
                if sensor in df.columns:
                    out_of_range = ((df[sensor] < min_val * 0.8) | 
                                  (df[sensor] > max_val * 1.2)).sum()
                    if out_of_range > len(df) * 0.1:  # More than 10% out of range
                        validation_results['issues'].append(
                            f"Sensor {sensor}: {out_of_range} values out of range"
                        )
                        validation_results['quality_score'] -= 0.05
            
            # Calculate statistics
            validation_results['statistics'] = {
                'row_count': len(df),
                'column_count': len(df.columns),
                'null_count': int(total_nulls),
                'duplicate_count': int(duplicates.sum()) if 'duplicates' in locals() else 0,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            # Ensure quality score is between 0 and 1
            validation_results['quality_score'] = max(0.0, min(1.0, validation_results['quality_score']))
            
            # Determine success based on quality score
            validation_results['success'] = validation_results['quality_score'] >= self.thresholds['min_quality_score']
            
        except Exception as e:
            logger.error(f"Error in pandas validation: {e}")
            validation_results['success'] = False
            validation_results['quality_score'] = 0.0
            validation_results['issues'].append(str(e))
        
        return validation_results
    
    def _parse_validation_results(self, results) -> Dict:
        """Parse Great Expectations validation results"""
        try:
            validation_summary = {
                'success': results.success,
                'quality_score': 0.0,
                'total_expectations': len(results.results),
                'successful_expectations': 0,
                'failed_expectations': 0,
                'issues': [],
                'statistics': {}
            }
            
            # Count successes and failures
            for result in results.results:
                if result.success:
                    validation_summary['successful_expectations'] += 1
                else:
                    validation_summary['failed_expectations'] += 1
                    # Add failure details
                    expectation_type = result.expectation_config.expectation_type
                    column = result.expectation_config.kwargs.get('column', 'N/A')
                    validation_summary['issues'].append(
                        f"{expectation_type} failed for column: {column}"
                    )
            
            # Calculate quality score
            if validation_summary['total_expectations'] > 0:
                validation_summary['quality_score'] = (
                    validation_summary['successful_expectations'] / 
                    validation_summary['total_expectations']
                )
            
            # Extract statistics if available
            if hasattr(results, 'statistics'):
                validation_summary['statistics'] = results.statistics
            
            return validation_summary
            
        except Exception as e:
            logger.error(f"Error parsing validation results: {e}")
            return {
                'success': False,
                'quality_score': 0.0,
                'issues': [str(e)]
            }
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive data quality report"""
        try:
            report = []
            report.append("=" * 60)
            report.append("DATA QUALITY REPORT")
            report.append("=" * 60)
            report.append(f"Generated: {datetime.utcnow().isoformat()}")
            report.append("")
            
            # Basic statistics
            report.append("DATASET OVERVIEW:")
            report.append(f"  - Total Records: {len(df):,}")
            report.append(f"  - Total Columns: {len(df.columns)}")
            report.append(f"  - Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            report.append("")
            
            # Column analysis
            report.append("COLUMN ANALYSIS:")
            for col in df.columns:
                dtype = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                null_pct = (null_count / len(df)) * 100
                unique_count = df[col].nunique()
                
                report.append(f"  {col}:")
                report.append(f"    - Type: {dtype}")
                report.append(f"    - Nulls: {null_count} ({null_pct:.2f}%)")
                report.append(f"    - Unique Values: {unique_count}")
                
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    report.append(f"    - Min: {df[col].min():.2f}")
                    report.append(f"    - Max: {df[col].max():.2f}")
                    report.append(f"    - Mean: {df[col].mean():.2f}")
                    report.append(f"    - Std: {df[col].std():.2f}")
            
            report.append("")
            
            # Data quality issues
            validation_results = self.validate_dataframe(df)
            
            report.append("DATA QUALITY ASSESSMENT:")
            report.append(f"  - Quality Score: {validation_results['quality_score']:.2f}")
            report.append(f"  - Status: {'✅ PASSED' if validation_results['success'] else '❌ FAILED'}")
            
            if validation_results['issues']:
                report.append("  - Issues Found:")
                for issue in validation_results['issues']:
                    report.append(f"    • {issue}")
            else:
                report.append("  - No issues found")
            
            report.append("")
            report.append("=" * 60)
            
            report_text = "\n".join(report)
            
            # Save report to file
            report_file = os.path.join("logs", f"data_quality_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt")
            os.makedirs("logs", exist_ok=True)
            
            with open(report_file, 'w') as f:
                f.write(report_text)
            
            logger.info(f"Data quality report saved to: {report_file}")
            
            return report_text
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {e}"

# =====================================================
# Utility Functions
# =====================================================

def validate_cmapss_data(file_path: str = None) -> Dict:
    """Validate C-MAPSS data file"""
    try:
        validator = DataQualityValidator()
        
        # Load data
        if file_path is None:
            file_path = os.path.join("data", "raw", "train_FD001.txt")
        
        # Read C-MAPSS data
        column_names = ['unit_number', 'time_cycles', 'altitude', 'mach_number', 'tra'] + \
                      [f's{i}' for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
        
        df = pd.read_csv(file_path, sep=' ', header=None, names=column_names)
        
        # Validate
        results = validator.validate_dataframe(df)
        
        # Generate report
        report = validator.generate_data_quality_report(df)
        print(report)
        
        return results
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {'success': False, 'quality_score': 0.0, 'issues': [str(e)]}

# =====================================================
# Main Function
# =====================================================

def main():
    """Main function for testing data quality validation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Testing Data Quality Validator...")
    
    # Create validator
    validator = DataQualityValidator()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'unit_number': [1, 1, 1, 2, 2, 2],
        'time_cycles': [1, 2, 3, 1, 2, 3],
        'altitude': [35000, 35000, 35000, 34000, 34000, 34000],
        'mach_number': [0.8, 0.8, 0.8, 0.75, 0.75, 0.75],
        'tra': [85, 85, 85, 84, 84, 84],
        's2': [640, 641, 642, 639, 640, 641],
        's3': [1600, 1601, 1602, 1599, 1600, 1601],
        's4': [1400, 1401, 1402, 1399, 1400, 1401],
        's7': [3.0, 3.0, 3.0, 2.9, 2.9, 2.9],
        's8': [9500, 9501, 9502, 9499, 9500, 9501],
        's9': [8750, 8751, 8752, 8749, 8750, 8751],
        's11': [3.0, 3.0, 3.0, 2.9, 2.9, 2.9],
        's12': [0.5, 0.5, 0.5, 0.49, 0.49, 0.49],
        's13': [9500, 9501, 9502, 9499, 9500, 9501],
        's14': [8750, 8751, 8752, 8749, 8750, 8751],
        's15': [5.5, 5.5, 5.5, 5.4, 5.4, 5.4],
        's17': [0.1, 0.1, 0.1, 0.09, 0.09, 0.09],
        's20': [0.1, 0.1, 0.1, 0.09, 0.09, 0.09],
        's21': [0.1, 0.1, 0.1, 0.09, 0.09, 0.09]
    })
    
    # Validate sample data
    results = validator.validate_dataframe(sample_data)
    
    print("\nValidation Results:")
    print(f"  Success: {results['success']}")
    print(f"  Quality Score: {results['quality_score']:.2f}")
    print(f"  Issues: {len(results.get('issues', []))}")
    
    # Generate report
    report = validator.generate_data_quality_report(sample_data)
    
    logger.info("✅ Data quality validator test completed!")
    
    return results['success']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)