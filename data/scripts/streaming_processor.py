"""
Streaming Data Processor
Handles real-time data processing and quality monitoring
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Deque
from collections import deque
import threading
import time

import pandas as pd
import numpy as np
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import get_config
from scripts.supabase_connector import SupabaseConnector
from scripts.data_quality_validator import DataQualityValidator
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
# Alert Models
# =====================================================

@dataclass
class Alert:
    """Data quality or system alert"""
    id: str
    timestamp: datetime
    engine_id: str
    alert_type: str  # 'data_quality', 'anomaly', 'rul_threshold', 'system'
    severity: str    # 'critical', 'high', 'medium', 'low'
    message: str
    details: Dict
    acknowledged: bool = False

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'engine_id': self.engine_id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'details': self.details,
            'acknowledged': self.acknowledged
        }

# =====================================================
# Streaming Processor Class
# =====================================================

class StreamingProcessor:
    """Real-time data processing and monitoring system"""
    
    def __init__(self):
        self.config = get_config()
        self.validator = DataQualityValidator()
        self.db_connector = None
        
        # Data buffers and windows
        self.data_window = deque(maxlen=1000)  # Rolling window of recent data
        self.quality_buffer = deque(maxlen=100)  # Quality metrics buffer
        self.alert_queue = deque(maxlen=500)  # Alert queue
        
        # Processing metrics
        self.metrics = {
            'total_readings': 0,
            'quality_violations': 0,
            'anomalies_detected': 0,
            'alerts_generated': 0,
            'processing_errors': 0,
            'avg_processing_time_ms': 0,
            'last_processed': None
        }
        
        # Thresholds
        self.thresholds = {
            'quality_score_min': 0.7,
            'anomaly_score_max': 75,
            'rul_critical': 10,
            'rul_warning': 30,
            'processing_time_max_ms': 100,
            'data_freshness_max_seconds': 30
        }
        
        # Callbacks for alerts
        self.alert_callbacks = []
        
        # Background processing
        self.processing_active = False
        self.background_tasks = []
        
        # Initialize database connection
        self._init_database()
    
    def _init_database(self):
        """Initialize database connection"""
        try:
            self.db_connector = SupabaseConnector(self.config.database)
            if not self.db_connector.test_connection():
                logger.warning("Database connection failed")
                self.db_connector = None
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")
            self.db_connector = None
    
    async def start_processing(self):
        """Start background processing tasks"""
        if self.processing_active:
            return
        
        self.processing_active = True
        logger.info("Starting streaming processor...")
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._quality_monitor()),
            asyncio.create_task(self._alert_processor()),
            asyncio.create_task(self._metrics_updater())
        ]
        
        logger.info("Background processing started")
    
    async def stop_processing(self):
        """Stop background processing tasks"""
        if not self.processing_active:
            return
        
        self.processing_active = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close database connection
        if self.db_connector:
            self.db_connector.close()
        
        logger.info("Streaming processor stopped")
    
    async def process_reading(self, reading: SensorReading) -> Dict:
        """Process a single sensor reading"""
        start_time = time.time()
        result = {
            'processed': False,
            'quality_score': 0.0,
            'alerts': [],
            'errors': []
        }
        
        try:
            # Convert to DataFrame for validation
            df = pd.DataFrame([reading.to_dict()])
            
            # Validate data quality
            validation_result = self.validator.validate_dataframe(df)
            result['quality_score'] = validation_result.get('quality_score', 0.0)
            
            # Add to data window
            self.data_window.append(reading)
            
            # Check quality thresholds
            if result['quality_score'] < self.thresholds['quality_score_min']:
                alert = self._create_alert(
                    reading.engine_id,
                    'data_quality',
                    'medium',
                    f"Low quality score: {result['quality_score']:.2f}",
                    {'quality_score': result['quality_score'], 'issues': validation_result.get('issues', [])}
                )
                result['alerts'].append(alert)
                self.metrics['quality_violations'] += 1
            
            # Check anomaly score
            if reading.anomaly_score > self.thresholds['anomaly_score_max']:
                alert = self._create_alert(
                    reading.engine_id,
                    'anomaly',
                    'high',
                    f"High anomaly score: {reading.anomaly_score:.1f}",
                    {'anomaly_score': reading.anomaly_score, 'sensor_readings': reading.to_dict()}
                )
                result['alerts'].append(alert)
                self.metrics['anomalies_detected'] += 1
            
            # Check RUL thresholds
            if reading.rul <= self.thresholds['rul_critical']:
                alert = self._create_alert(
                    reading.engine_id,
                    'rul_threshold',
                    'critical',
                    f"Critical RUL: {reading.rul} cycles remaining",
                    {'rul': reading.rul, 'health_score': reading.health_score}
                )
                result['alerts'].append(alert)
            elif reading.rul <= self.thresholds['rul_warning']:
                alert = self._create_alert(
                    reading.engine_id,
                    'rul_threshold',
                    'medium',
                    f"Warning RUL: {reading.rul} cycles remaining",
                    {'rul': reading.rul, 'health_score': reading.health_score}
                )
                result['alerts'].append(alert)
            
            # Add alerts to queue
            for alert in result['alerts']:
                self.alert_queue.append(alert)
                self.metrics['alerts_generated'] += 1
            
            # Update metrics
            self.metrics['total_readings'] += 1
            processing_time_ms = (time.time() - start_time) * 1000
            self.metrics['last_processed'] = datetime.utcnow()
            
            # Update average processing time
            if self.metrics['avg_processing_time_ms'] == 0:
                self.metrics['avg_processing_time_ms'] = processing_time_ms
            else:
                self.metrics['avg_processing_time_ms'] = (
                    self.metrics['avg_processing_time_ms'] * 0.9 + processing_time_ms * 0.1
                )
            
            # Check processing time threshold
            if processing_time_ms > self.thresholds['processing_time_max_ms']:
                alert = self._create_alert(
                    'system',
                    'system',
                    'medium',
                    f"Slow processing: {processing_time_ms:.1f}ms",
                    {'processing_time_ms': processing_time_ms}
                )
                result['alerts'].append(alert)
                self.alert_queue.append(alert)
            
            result['processed'] = True
            
        except Exception as e:
            logger.error(f"Error processing reading: {e}")
            result['errors'].append(str(e))
            self.metrics['processing_errors'] += 1
        
        return result
    
    def _create_alert(self, engine_id: str, alert_type: str, severity: str, 
                     message: str, details: Dict) -> Alert:
        """Create a new alert"""
        alert_id = f"{alert_type}_{engine_id}_{int(time.time() * 1000)}"
        
        return Alert(
            id=alert_id,
            timestamp=datetime.utcnow(),
            engine_id=engine_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details
        )
    
    async def _quality_monitor(self):
        """Background task to monitor data quality"""
        while self.processing_active:
            try:
                if len(self.data_window) < 10:
                    await asyncio.sleep(5)
                    continue
                
                # Convert recent data to DataFrame
                recent_data = list(self.data_window)[-100:]  # Last 100 readings
                df = pd.DataFrame([r.to_dict() for r in recent_data])
                
                # Calculate quality metrics
                quality_metrics = {
                    'timestamp': datetime.utcnow(),
                    'sample_size': len(df),
                    'avg_quality_score': df.get('health_score', pd.Series([0])).mean(),
                    'avg_anomaly_score': df.get('anomaly_score', pd.Series([0])).mean(),
                    'engines_monitored': df['unit_number'].nunique(),
                    'null_percentage': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
                    'duplicate_count': df.duplicated().sum()
                }
                
                self.quality_buffer.append(quality_metrics)
                
                # Log quality metrics
                if len(self.quality_buffer) % 10 == 0:  # Log every 10th iteration
                    logger.info(f"Quality metrics: "
                              f"Avg Quality={quality_metrics['avg_quality_score']:.1f}, "
                              f"Avg Anomaly={quality_metrics['avg_anomaly_score']:.1f}, "
                              f"Engines={quality_metrics['engines_monitored']}")
                
                # Check for trends
                if len(self.quality_buffer) >= 5:
                    await self._check_quality_trends()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Quality monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _check_quality_trends(self):
        """Check for concerning quality trends"""
        if len(self.quality_buffer) < 5:
            return
        
        recent_metrics = list(self.quality_buffer)[-5:]
        
        # Check for declining quality trend
        quality_scores = [m['avg_quality_score'] for m in recent_metrics]
        quality_trend = np.polyfit(range(len(quality_scores)), quality_scores, 1)[0]
        
        if quality_trend < -2:  # Declining by more than 2 points per measurement
            alert = self._create_alert(
                'system',
                'data_quality',
                'high',
                f"Declining quality trend detected: {quality_trend:.2f}",
                {'trend': quality_trend, 'recent_scores': quality_scores}
            )
            self.alert_queue.append(alert)
        
        # Check for increasing anomaly trend
        anomaly_scores = [m['avg_anomaly_score'] for m in recent_metrics]
        anomaly_trend = np.polyfit(range(len(anomaly_scores)), anomaly_scores, 1)[0]
        
        if anomaly_trend > 5:  # Increasing by more than 5 points per measurement
            alert = self._create_alert(
                'system',
                'anomaly',
                'high',
                f"Increasing anomaly trend detected: {anomaly_trend:.2f}",
                {'trend': anomaly_trend, 'recent_scores': anomaly_scores}
            )
            self.alert_queue.append(alert)
    
    async def _alert_processor(self):
        """Background task to process alerts"""
        while self.processing_active:
            try:
                if not self.alert_queue:
                    await asyncio.sleep(1)
                    continue
                
                # Process alerts in batches
                alerts_to_process = []
                while self.alert_queue and len(alerts_to_process) < 10:
                    alerts_to_process.append(self.alert_queue.popleft())
                
                # Send alerts to database
                if self.db_connector and alerts_to_process:
                    await self._send_alerts_to_database(alerts_to_process)
                
                # Trigger alert callbacks
                for alert in alerts_to_process:
                    for callback in self.alert_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(alert)
                            else:
                                callback(alert)
                        except Exception as e:
                            logger.error(f"Alert callback error: {e}")
                
                # Log critical alerts
                critical_alerts = [a for a in alerts_to_process if a.severity == 'critical']
                for alert in critical_alerts:
                    logger.error(f"🚨 CRITICAL ALERT: {alert.message} - Engine: {alert.engine_id}")
                
                await asyncio.sleep(0.1)  # Small delay between batches
                
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(1)
    
    async def _send_alerts_to_database(self, alerts: List[Alert]):
        """Send alerts to database"""
        try:
            # Convert alerts to DataFrame
            alert_data = []
            for alert in alerts:
                alert_dict = alert.to_dict()
                alert_dict['details'] = json.dumps(alert_dict['details'])
                alert_data.append({
                    'engine_id': None,  # Will be mapped from engine name
                    'unit_number': int(alert.engine_id.split('_')[-1]) if '_' in alert.engine_id else 0,
                    'alert_type': alert.alert_type,
                    'severity': alert.severity,
                    'message': alert.message,
                    'details': alert_dict['details'],
                    'is_acknowledged': alert.acknowledged
                })
            
            df = pd.DataFrame(alert_data)
            
            # Insert using Supabase client
            for _, row in df.iterrows():
                try:
                    # Get or create engine
                    engine = self.db_connector.get_or_create_engine(row['unit_number'])
                    row['engine_id'] = engine['id']
                    
                    # Insert alert
                    self.db_connector.client.table('alerts').insert(row.to_dict()).execute()
                except Exception as e:
                    logger.error(f"Failed to insert alert: {e}")
            
            logger.info(f"Sent {len(alerts)} alerts to database")
            
        except Exception as e:
            logger.error(f"Failed to send alerts to database: {e}")
    
    async def _metrics_updater(self):
        """Background task to update and log metrics"""
        while self.processing_active:
            try:
                # Log metrics periodically
                if self.metrics['total_readings'] > 0:
                    logger.info(f"Processing metrics: "
                              f"Readings={self.metrics['total_readings']}, "
                              f"Errors={self.metrics['processing_errors']}, "
                              f"Alerts={self.metrics['alerts_generated']}, "
                              f"Avg Time={self.metrics['avg_processing_time_ms']:.1f}ms")
                
                # Check data freshness
                if self.metrics['last_processed']:
                    time_since_last = (datetime.utcnow() - self.metrics['last_processed']).total_seconds()
                    if time_since_last > self.thresholds['data_freshness_max_seconds']:
                        alert = self._create_alert(
                            'system',
                            'system',
                            'high',
                            f"No data received for {time_since_last:.0f} seconds",
                            {'seconds_since_last': time_since_last}
                        )
                        self.alert_queue.append(alert)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Metrics update error: {e}")
                await asyncio.sleep(10)
    
    def register_alert_callback(self, callback: Callable):
        """Register a callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def get_metrics(self) -> Dict:
        """Get current processing metrics"""
        return {
            **self.metrics,
            'quality_buffer_size': len(self.quality_buffer),
            'data_window_size': len(self.data_window),
            'pending_alerts': len(self.alert_queue)
        }
    
    def get_recent_quality_metrics(self, limit: int = 10) -> List[Dict]:
        """Get recent quality metrics"""
        recent = list(self.quality_buffer)[-limit:]
        return [
            {
                **metrics,
                'timestamp': metrics['timestamp'].isoformat()
            }
            for metrics in recent
        ]
    
    def get_pending_alerts(self, limit: int = 50) -> List[Dict]:
        """Get pending alerts"""
        alerts = list(self.alert_queue)[-limit:]
        return [alert.to_dict() for alert in alerts]

# =====================================================
# Utility Functions
# =====================================================

async def demo_processor():
    """Demo streaming processor with simulated data"""
    from sensor_simulator import SensorSimulator
    
    # Alert callback
    def on_alert(alert: Alert):
        severity_emoji = {'critical': '🔥', 'high': '⚠️', 'medium': '⚡', 'low': '💡'}
        emoji = severity_emoji.get(alert.severity, '📋')
        print(f"{emoji} ALERT [{alert.severity.upper()}]: {alert.message} (Engine: {alert.engine_id})")
    
    # Create processor and simulator
    processor = StreamingProcessor()
    processor.register_alert_callback(on_alert)
    
    simulator = SensorSimulator(num_engines=2, frequency_hz=2.0)
    
    # Register simulator callback to send data to processor
    async def process_reading(reading):
        await processor.process_reading(reading)
    
    simulator.register_callback('on_reading', process_reading)
    
    # Start processor
    await processor.start_processing()
    
    # Inject some anomalies for demo
    async def inject_demo_anomalies():
        await asyncio.sleep(5)
        simulator.inject_anomaly(1, 'spike')
        await asyncio.sleep(10)
        simulator.inject_anomaly(2, 'failure')
    
    # Run demo
    await asyncio.gather(
        simulator.start(duration_seconds=30),
        inject_demo_anomalies()
    )
    
    # Stop processor
    await processor.stop_processing()
    
    # Print final metrics
    print("\nFinal Metrics:")
    print(json.dumps(processor.get_metrics(), indent=2, default=str))

# =====================================================
# Main Function
# =====================================================

def main():
    """Main function for streaming processor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Streaming Data Processor')
    parser.add_argument('--demo', action='store_true', help='Run demo mode')
    
    args = parser.parse_args()
    
    if args.demo:
        # Run demo
        asyncio.run(demo_processor())
    else:
        # Run standalone processor (for external data sources)
        processor = StreamingProcessor()
        
        try:
            asyncio.run(processor.start_processing())
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            asyncio.run(processor.stop_processing())

if __name__ == "__main__":
    main()