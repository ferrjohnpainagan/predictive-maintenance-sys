"""
Real-time Data Quality Validation Service
Continuous monitoring and validation of streaming sensor data
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from collections import deque, defaultdict
import time
import threading

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import get_config
from scripts.data_quality_validator import DataQualityValidator
from scripts.cloudwatch_monitor import send_pipeline_metrics_to_cloudwatch
from scripts.sensor_simulator import SensorReading

# =====================================================
# Logging Setup
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================
# Validation Models
# =====================================================

@dataclass
class ValidationResult:
    """Real-time validation result"""
    reading_id: str
    timestamp: datetime
    engine_id: str
    passed: bool
    quality_score: float
    issues: List[str]
    severity: str  # 'low', 'medium', 'high', 'critical'
    response_time_ms: float

@dataclass
class QualityMetrics:
    """Quality metrics for monitoring"""
    window_start: datetime
    window_end: datetime
    total_readings: int
    passed_readings: int
    failed_readings: int
    avg_quality_score: float
    avg_response_time_ms: float
    error_rate_percent: float
    engines_monitored: int
    top_issues: List[str]

# =====================================================
# Real-time Validator Class
# =====================================================

class RealtimeValidator:
    """Real-time data quality validation service"""
    
    def __init__(self, window_size: int = 100, validation_interval: float = 1.0):
        self.config = get_config()
        self.base_validator = DataQualityValidator()
        
        # Configuration
        self.window_size = window_size
        self.validation_interval = validation_interval
        
        # Data structures
        self.validation_window = deque(maxlen=window_size)
        self.quality_history = deque(maxlen=1000)  # Store quality metrics
        self.engine_stats = defaultdict(lambda: {
            'readings_count': 0,
            'quality_scores': deque(maxlen=50),
            'recent_issues': deque(maxlen=20),
            'last_validation': None
        })
        
        # Validation rules and thresholds (relaxed for development)
        self.validation_rules = {
            'sensor_ranges': self.config.data_processing.sensor_ranges,
            'quality_threshold': 0.5,  # Relaxed from 0.7
            'anomaly_threshold': 85,   # Relaxed from 75
            'rul_critical': 5,         # More strict critical level  
            'rul_warning': 20,         # Relaxed warning level
            'response_time_max': 100,  # Relaxed response time
            'null_tolerance': 0.1,     # Relaxed null tolerance
            'duplicate_tolerance': 0.05 # Relaxed duplicate tolerance
        }
        
        # Real-time metrics
        self.metrics = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'avg_response_time_ms': 0,
            'current_quality_score': 0,
            'alerts_triggered': 0,
            'engines_monitored': 0,
            'validations_per_second': 0
        }
        
        # Callbacks
        self.validation_callbacks = []
        self.alert_callbacks = []
        
        # Background processing
        self.running = False
        self.background_tasks = []
        
        # Performance tracking
        self.validation_times = deque(maxlen=100)
        self.last_metrics_update = time.time()
        self.validations_in_last_second = 0
        
        logger.info("Real-time validator initialized")
    
    async def start_validation_service(self):
        """Start the real-time validation service"""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting real-time validation service...")
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._quality_monitor()),
            asyncio.create_task(self._metrics_updater()),
            asyncio.create_task(self._cloudwatch_reporter())
        ]
        
        logger.info("Real-time validation service started")
    
    async def stop_validation_service(self):
        """Stop the validation service"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("Real-time validation service stopped")
    
    async def validate_reading(self, reading: SensorReading) -> ValidationResult:
        """Validate a single sensor reading in real-time"""
        start_time = time.time()
        reading_id = f"{reading.engine_id}_{reading.time_cycles}_{int(time.time() * 1000)}"
        
        issues = []
        quality_score = 100.0
        severity = 'low'
        
        try:
            # Convert to DataFrame for validation
            df = pd.DataFrame([reading.to_dict()])
            
            # Run core validation
            base_result = self.base_validator.validate_dataframe(df)
            quality_score = base_result.get('quality_score', 0) * 100
            
            # Real-time specific validations
            await self._validate_sensor_ranges(reading, issues)
            await self._validate_operational_parameters(reading, issues)
            await self._validate_temporal_consistency(reading, issues)
            await self._validate_cross_correlations(reading, issues)
            
            # Determine severity
            severity = self._calculate_severity(issues, quality_score, reading)
            
            # Adjust quality score based on issues
            if issues:
                quality_score *= (1 - len(issues) * 0.1)  # Reduce score by 10% per issue
                quality_score = max(0, quality_score)
            
            # Create validation result
            response_time_ms = (time.time() - start_time) * 1000
            result = ValidationResult(
                reading_id=reading_id,
                timestamp=datetime.utcnow(),
                engine_id=reading.engine_id,
                passed=quality_score >= self.validation_rules['quality_threshold'] * 100,
                quality_score=quality_score,
                issues=issues,
                severity=severity,
                response_time_ms=response_time_ms
            )
            
            # Update metrics and statistics
            await self._update_validation_metrics(result, reading)
            
            # Trigger callbacks
            for callback in self.validation_callbacks:
                await self._safe_callback(callback, result)
            
            # Trigger alerts if necessary
            if not result.passed or severity in ['high', 'critical']:
                await self._trigger_alert(result, reading)
            
            return result
            
        except Exception as e:
            logger.error(f"Validation error for reading {reading_id}: {e}")
            response_time_ms = (time.time() - start_time) * 1000
            
            return ValidationResult(
                reading_id=reading_id,
                timestamp=datetime.utcnow(),
                engine_id=reading.engine_id,
                passed=False,
                quality_score=0.0,
                issues=[f"Validation error: {str(e)}"],
                severity='critical',
                response_time_ms=response_time_ms
            )
    
    async def _validate_sensor_ranges(self, reading: SensorReading, issues: List[str]):
        """Validate sensor values are within expected ranges (with tolerance)"""
        tolerance = 0.3  # 30% tolerance for development
        
        for sensor, (min_val, max_val) in self.validation_rules['sensor_ranges'].items():
            if hasattr(reading, sensor):
                value = getattr(reading, sensor)
                if value is not None:
                    # Add tolerance to ranges
                    range_size = max_val - min_val
                    tolerant_min = min_val - (range_size * tolerance)
                    tolerant_max = max_val + (range_size * tolerance)
                    
                    if value < tolerant_min or value > tolerant_max:
                        issues.append(f"Sensor {sensor} significantly out of range: {value:.2f} (expected {min_val}-{max_val} ±{tolerance*100:.0f}%)")
    
    async def _validate_operational_parameters(self, reading: SensorReading, issues: List[str]):
        """Validate operational parameters"""
        # Altitude check
        if reading.altitude < 0 or reading.altitude > 50000:
            issues.append(f"Invalid altitude: {reading.altitude}")
        
        # Mach number check
        if reading.mach_number < 0 or reading.mach_number > 1.5:
            issues.append(f"Invalid Mach number: {reading.mach_number}")
        
        # TRA check
        if reading.tra < 0 or reading.tra > 100:
            issues.append(f"Invalid TRA: {reading.tra}")
        
        # RUL check
        if reading.rul < 0 or reading.rul > 200:
            issues.append(f"Invalid RUL: {reading.rul}")
        
        # Health score check
        if reading.health_score < 0 or reading.health_score > 100:
            issues.append(f"Invalid health score: {reading.health_score}")
    
    async def _validate_temporal_consistency(self, reading: SensorReading, issues: List[str]):
        """Validate temporal consistency with previous readings"""
        engine_stats = self.engine_stats[reading.engine_id]
        
        # Check time cycles progression
        if engine_stats['last_validation']:
            last_reading = engine_stats['last_validation']
            if reading.time_cycles <= last_reading.time_cycles:
                issues.append(f"Time cycles not progressing: {reading.time_cycles} <= {last_reading.time_cycles}")
        
        # Update last validation
        engine_stats['last_validation'] = reading
    
    async def _validate_cross_correlations(self, reading: SensorReading, issues: List[str]):
        """Validate cross-correlations between sensors"""
        # Example: Fan speed (s8) and corrected fan speed (s13) should be correlated
        if hasattr(reading, 's8') and hasattr(reading, 's13'):
            if reading.s8 > 0 and reading.s13 > 0:
                ratio = reading.s13 / reading.s8
                if ratio < 0.8 or ratio > 1.2:  # Expected correlation
                    issues.append(f"Abnormal fan speed correlation: s13/s8 = {ratio:.2f}")
        
        # Temperature consistency checks
        temp_sensors = ['s2', 's3', 's4']
        temps = [getattr(reading, sensor) for sensor in temp_sensors if hasattr(reading, sensor)]
        if len(temps) >= 2:
            # Check for unrealistic temperature differences
            max_diff = max(temps) - min(temps)
            if max_diff > 1000:  # Unrealistic temperature difference
                issues.append(f"Large temperature variation: {max_diff:.1f}°R")
    
    def _calculate_severity(self, issues: List[str], quality_score: float, reading: SensorReading) -> str:
        """Calculate issue severity based on validation results"""
        # Critical conditions
        if reading.rul <= self.validation_rules['rul_critical']:
            return 'critical'
        
        if quality_score < 30:
            return 'critical'
        
        # High severity conditions
        if reading.anomaly_score > self.validation_rules['anomaly_threshold']:
            return 'high'
        
        if len(issues) >= 3:
            return 'high'
        
        if quality_score < 50:
            return 'high'
        
        # Medium severity conditions
        if reading.rul <= self.validation_rules['rul_warning']:
            return 'medium'
        
        if len(issues) >= 1:
            return 'medium'
        
        if quality_score < 70:
            return 'medium'
        
        return 'low'
    
    async def _update_validation_metrics(self, result: ValidationResult, reading: SensorReading):
        """Update validation metrics and statistics"""
        # Update global metrics
        self.metrics['total_validations'] += 1
        if result.passed:
            self.metrics['successful_validations'] += 1
        else:
            self.metrics['failed_validations'] += 1
        
        # Update response time
        self.validation_times.append(result.response_time_ms)
        self.metrics['avg_response_time_ms'] = np.mean(self.validation_times)
        
        # Update quality score
        self.metrics['current_quality_score'] = result.quality_score
        
        # Count alerts
        if result.severity in ['high', 'critical']:
            self.metrics['alerts_triggered'] += 1
        
        # Update engine statistics
        engine_stats = self.engine_stats[reading.engine_id]
        engine_stats['readings_count'] += 1
        engine_stats['quality_scores'].append(result.quality_score)
        
        if result.issues:
            engine_stats['recent_issues'].extend(result.issues)
        
        # Update engines monitored count
        self.metrics['engines_monitored'] = len(self.engine_stats)
        
        # Add to validation window
        self.validation_window.append(result)
    
    async def _trigger_alert(self, result: ValidationResult, reading: SensorReading):
        """Trigger alert for failed validation or high severity issues"""
        alert_data = {
            'timestamp': result.timestamp.isoformat(),
            'engine_id': result.engine_id,
            'reading_id': result.reading_id,
            'severity': result.severity,
            'quality_score': result.quality_score,
            'issues': result.issues,
            'sensor_data': asdict(reading)
        }
        
        # Log alert
        severity_emoji = {'critical': '🔥', 'high': '⚠️', 'medium': '⚡', 'low': '💡'}
        emoji = severity_emoji.get(result.severity, '📋')
        
        logger.warning(f"{emoji} VALIDATION ALERT [{result.severity.upper()}]: "
                      f"Engine {result.engine_id}, Quality: {result.quality_score:.1f}%, "
                      f"Issues: {len(result.issues)}")
        
        # Trigger alert callbacks
        for callback in self.alert_callbacks:
            await self._safe_callback(callback, alert_data)
    
    async def _safe_callback(self, callback: Callable, data):
        """Execute callback safely"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
        except Exception as e:
            logger.error(f"Callback error: {e}")
    
    async def _quality_monitor(self):
        """Background task to monitor overall quality trends"""
        while self.running:
            try:
                if len(self.validation_window) >= 10:
                    # Calculate quality metrics for current window
                    recent_validations = list(self.validation_window)
                    
                    metrics = QualityMetrics(
                        window_start=recent_validations[0].timestamp,
                        window_end=recent_validations[-1].timestamp,
                        total_readings=len(recent_validations),
                        passed_readings=sum(1 for v in recent_validations if v.passed),
                        failed_readings=sum(1 for v in recent_validations if not v.passed),
                        avg_quality_score=np.mean([v.quality_score for v in recent_validations]),
                        avg_response_time_ms=np.mean([v.response_time_ms for v in recent_validations]),
                        error_rate_percent=0,
                        engines_monitored=len(set(v.engine_id for v in recent_validations)),
                        top_issues=[]
                    )
                    
                    # Calculate error rate
                    if metrics.total_readings > 0:
                        metrics.error_rate_percent = (metrics.failed_readings / metrics.total_readings) * 100
                    
                    # Get top issues
                    all_issues = []
                    for v in recent_validations:
                        all_issues.extend(v.issues)
                    
                    if all_issues:
                        from collections import Counter
                        issue_counts = Counter(all_issues)
                        metrics.top_issues = [issue for issue, count in issue_counts.most_common(5)]
                    
                    # Store metrics
                    self.quality_history.append(metrics)
                    
                    # Log quality summary
                    if len(self.quality_history) % 10 == 0:  # Log every 10th iteration
                        logger.info(f"Quality Monitor: "
                                  f"Score={metrics.avg_quality_score:.1f}%, "
                                  f"Error Rate={metrics.error_rate_percent:.1f}%, "
                                  f"Response Time={metrics.avg_response_time_ms:.1f}ms")
                
                await asyncio.sleep(self.validation_interval * 10)  # Check every 10 intervals
                
            except Exception as e:
                logger.error(f"Quality monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _metrics_updater(self):
        """Background task to update performance metrics"""
        while self.running:
            try:
                current_time = time.time()
                time_elapsed = current_time - self.last_metrics_update
                
                if time_elapsed >= 1.0:  # Update every second
                    # Calculate validations per second
                    validations_delta = self.metrics['total_validations'] - self.validations_in_last_second
                    self.metrics['validations_per_second'] = validations_delta / time_elapsed
                    
                    self.validations_in_last_second = self.metrics['total_validations']
                    self.last_metrics_update = current_time
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Metrics updater error: {e}")
                await asyncio.sleep(5)
    
    async def _cloudwatch_reporter(self):
        """Background task to send metrics to CloudWatch"""
        while self.running:
            try:
                if self.metrics['total_validations'] > 0:
                    # Prepare CloudWatch metrics
                    cloudwatch_metrics = {
                        'data_quality_score': self.metrics['current_quality_score'],
                        'total_records_processed': self.metrics['total_validations'],
                        'processing_time_ms': self.metrics['avg_response_time_ms'],
                        'error_count': self.metrics['failed_validations'],
                        'anomaly_count': self.metrics['alerts_triggered'],
                        'success': True
                    }
                    
                    # Send to CloudWatch
                    success = send_pipeline_metrics_to_cloudwatch(cloudwatch_metrics)
                    if success:
                        logger.debug("Sent metrics to CloudWatch")
                
                await asyncio.sleep(60)  # Send every minute
                
            except Exception as e:
                logger.error(f"CloudWatch reporter error: {e}")
                await asyncio.sleep(30)
    
    def register_validation_callback(self, callback: Callable):
        """Register callback for validation results"""
        self.validation_callbacks.append(callback)
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_metrics(self) -> Dict:
        """Get current validation metrics"""
        return {
            **self.metrics,
            'validation_window_size': len(self.validation_window),
            'quality_history_size': len(self.quality_history),
            'engines_monitored_detail': {
                engine_id: {
                    'readings_count': stats['readings_count'],
                    'avg_quality_score': np.mean(stats['quality_scores']) if stats['quality_scores'] else 0,
                    'recent_issues_count': len(stats['recent_issues'])
                }
                for engine_id, stats in self.engine_stats.items()
            }
        }
    
    def get_quality_summary(self, limit: int = 10) -> List[Dict]:
        """Get recent quality metrics summary"""
        recent = list(self.quality_history)[-limit:]
        return [asdict(metrics) for metrics in recent]

# =====================================================
# Utility Functions
# =====================================================

async def demo_realtime_validation():
    """Demo real-time validation with sensor simulator"""
    from sensor_simulator import SensorSimulator
    
    # Create validator and simulator
    validator = RealtimeValidator(window_size=50, validation_interval=0.5)
    simulator = SensorSimulator(num_engines=2, frequency_hz=3.0)
    
    # Callbacks
    async def on_validation_result(result: ValidationResult):
        if not result.passed:
            print(f"❌ VALIDATION FAILED: Engine {result.engine_id}, "
                  f"Score: {result.quality_score:.1f}%, Issues: {len(result.issues)}")
    
    async def on_alert(alert_data: Dict):
        print(f"🚨 ALERT: {alert_data['severity'].upper()} - Engine {alert_data['engine_id']}")
    
    # Register callbacks
    validator.register_validation_callback(on_validation_result)
    validator.register_alert_callback(on_alert)
    
    # Connect simulator to validator
    async def validate_reading(reading):
        await validator.validate_reading(reading)
    
    simulator.register_callback('on_reading', validate_reading)
    
    # Start services
    await validator.start_validation_service()
    
    # Inject anomalies for demo
    async def inject_demo_issues():
        await asyncio.sleep(3)
        simulator.inject_anomaly(1, 'spike')
        await asyncio.sleep(5)
        simulator.inject_anomaly(2, 'failure')
    
    # Run demo
    await asyncio.gather(
        simulator.start(duration_seconds=20),
        inject_demo_issues()
    )
    
    # Stop validator
    await validator.stop_validation_service()
    
    # Print final metrics
    print("\nValidation Metrics:")
    print(json.dumps(validator.get_metrics(), indent=2, default=str))

# =====================================================
# Main Function
# =====================================================

def main():
    """Main function for real-time validator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Data Quality Validator')
    parser.add_argument('--demo', action='store_true', help='Run demo mode')
    parser.add_argument('--window-size', type=int, default=100, help='Validation window size')
    parser.add_argument('--interval', type=float, default=1.0, help='Validation interval')
    
    args = parser.parse_args()
    
    if args.demo:
        # Run demo
        asyncio.run(demo_realtime_validation())
    else:
        # Run standalone validator
        validator = RealtimeValidator(
            window_size=args.window_size,
            validation_interval=args.interval
        )
        
        try:
            asyncio.run(validator.start_validation_service())
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            asyncio.run(validator.stop_validation_service())

if __name__ == "__main__":
    main()