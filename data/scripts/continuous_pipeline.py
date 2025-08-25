#!/usr/bin/env python3
"""
Continuous Data Pipeline Orchestrator
Integrates real-time simulation, streaming processing, validation, and monitoring
"""

import asyncio
import json
import logging
import os
import sys
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import get_config
from scripts.sensor_simulator import SensorSimulator, SensorReading
from scripts.streaming_processor import StreamingProcessor
from scripts.realtime_validator import RealtimeValidator
from scripts.cloudwatch_monitor import CloudWatchMonitor

# =====================================================
# Logging Setup
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/continuous_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# =====================================================
# Pipeline Orchestrator
# =====================================================

class ContinuousPipelineOrchestrator:
    """Orchestrates the complete continuous data pipeline"""
    
    def __init__(self, config: Dict):
        self.config = get_config()
        self.pipeline_config = config
        
        # Components
        self.simulator = None
        self.processor = None
        self.validator = None
        self.cloudwatch = None
        
        # Pipeline state
        self.running = False
        self.start_time = None
        self.components_health = {
            'simulator': 'stopped',
            'processor': 'stopped', 
            'validator': 'stopped',
            'cloudwatch': 'stopped'
        }
        
        # Performance metrics
        self.pipeline_metrics = {
            'total_readings_processed': 0,
            'quality_validations': 0,
            'alerts_generated': 0,
            'errors_encountered': 0,
            'uptime_seconds': 0,
            'avg_throughput': 0
        }
        
        # Graceful shutdown handling
        self.shutdown_event = asyncio.Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Continuous pipeline orchestrator initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.shutdown())
    
    async def initialize_components(self):
        """Initialize all pipeline components"""
        logger.info("Initializing pipeline components...")
        
        try:
            # Initialize sensor simulator
            self.simulator = SensorSimulator(
                num_engines=self.pipeline_config.get('num_engines', 5),
                frequency_hz=self.pipeline_config.get('frequency_hz', 2.0)
            )
            self.components_health['simulator'] = 'initialized'
            
            # Initialize streaming processor
            self.processor = StreamingProcessor()
            self.components_health['processor'] = 'initialized'
            
            # Initialize real-time validator
            self.validator = RealtimeValidator(
                window_size=self.pipeline_config.get('validation_window_size', 100),
                validation_interval=self.pipeline_config.get('validation_interval', 1.0)
            )
            self.components_health['validator'] = 'initialized'
            
            # Initialize CloudWatch monitor (optional)
            if self.pipeline_config.get('enable_cloudwatch', False):
                try:
                    self.cloudwatch = CloudWatchMonitor()
                    self.components_health['cloudwatch'] = 'initialized'
                except Exception as e:
                    logger.warning(f"CloudWatch initialization failed: {e}")
                    self.cloudwatch = None
                    self.components_health['cloudwatch'] = 'failed'
            
            # Wire components together
            await self._wire_components()
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            return False
    
    async def _wire_components(self):
        """Wire components together with callbacks"""
        # Sensor readings flow: Simulator -> Processor -> Validator
        
        # Connect simulator to processor and validator
        async def process_and_validate_reading(reading: SensorReading):
            try:
                # Process reading
                processing_result = await self.processor.process_reading(reading)
                
                # Validate reading
                validation_result = await self.validator.validate_reading(reading)
                
                # Update pipeline metrics
                self.pipeline_metrics['total_readings_processed'] += 1
                self.pipeline_metrics['quality_validations'] += 1
                
                if not validation_result.passed:
                    self.pipeline_metrics['errors_encountered'] += 1
                
                # Log occasional status updates
                if self.pipeline_metrics['total_readings_processed'] % 10 == 0:
                    success_rate = ((self.pipeline_metrics['quality_validations'] - self.pipeline_metrics['errors_encountered']) / 
                                  max(1, self.pipeline_metrics['quality_validations'])) * 100
                    logger.info(f"📊 Processed {self.pipeline_metrics['total_readings_processed']} readings, "
                              f"success rate: {success_rate:.1f}%")
                
            except Exception as e:
                logger.error(f"Error processing reading: {e}")
                self.pipeline_metrics['errors_encountered'] += 1
        
        self.simulator.register_callback('on_reading', process_and_validate_reading)
        
        # Connect processor alerts to logging
        def log_processor_alert(alert):
            logger.warning(f"PROCESSOR ALERT: {alert.message} - Engine: {alert.engine_id}")
            self.pipeline_metrics['alerts_generated'] += 1
        
        self.processor.register_alert_callback(log_processor_alert)
        
        # Connect validator alerts to logging
        async def log_validator_alert(alert_data):
            logger.warning(f"VALIDATOR ALERT: {alert_data['severity']} - Engine: {alert_data['engine_id']}")
            self.pipeline_metrics['alerts_generated'] += 1
        
        self.validator.register_alert_callback(log_validator_alert)
        
        # Connect batch processing to CloudWatch
        if self.cloudwatch:
            async def send_batch_metrics(df):
                try:
                    # Calculate metrics from batch
                    metrics = {
                        'total_records_processed': len(df),
                        'processing_time_ms': 100,  # Estimated
                        'success': True
                    }
                    self.cloudwatch.send_pipeline_metrics(metrics)
                except Exception as e:
                    logger.error(f"Failed to send CloudWatch metrics: {e}")
            
            self.simulator.register_callback('on_batch_ready', send_batch_metrics)
    
    async def start_pipeline(self, duration_seconds: Optional[int] = None):
        """Start the continuous pipeline"""
        if self.running:
            logger.warning("Pipeline is already running")
            return
        
        logger.info("=" * 60)
        logger.info("STARTING CONTINUOUS DATA PIPELINE")
        logger.info("=" * 60)
        
        self.running = True
        self.start_time = datetime.utcnow()
        
        try:
            # Start all components
            await self._start_components()
            
            # Start monitoring task
            monitoring_task = asyncio.create_task(self._monitor_pipeline())
            
            # Start simulator (this will drive the entire pipeline)
            logger.info(f"Starting data generation for {duration_seconds or 'indefinite'} seconds...")
            
            simulator_task = asyncio.create_task(
                self.simulator.start(duration_seconds=duration_seconds)
            )
            
            # Wait for completion or shutdown
            try:
                await asyncio.gather(simulator_task, monitoring_task)
            except asyncio.CancelledError:
                logger.info("Pipeline tasks cancelled")
            
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def _start_components(self):
        """Start all pipeline components"""
        # Start streaming processor
        await self.processor.start_processing()
        self.components_health['processor'] = 'running'
        logger.info("✅ Streaming processor started")
        
        # Start real-time validator
        await self.validator.start_validation_service()
        self.components_health['validator'] = 'running'
        logger.info("✅ Real-time validator started")
        
        # CloudWatch is passive, just mark as running
        if self.cloudwatch:
            self.components_health['cloudwatch'] = 'running'
            logger.info("✅ CloudWatch monitor ready")
        
        self.components_health['simulator'] = 'running'
        logger.info("✅ All components started successfully")
    
    async def _monitor_pipeline(self):
        """Monitor pipeline health and performance"""
        logger.info("Pipeline monitoring started")
        
        while self.running:
            try:
                # Update uptime
                if self.start_time:
                    self.pipeline_metrics['uptime_seconds'] = (
                        datetime.utcnow() - self.start_time
                    ).total_seconds()
                
                # Calculate throughput
                if self.pipeline_metrics['uptime_seconds'] > 0:
                    self.pipeline_metrics['avg_throughput'] = (
                        self.pipeline_metrics['total_readings_processed'] / 
                        self.pipeline_metrics['uptime_seconds']
                    )
                
                # Log health status every 30 seconds
                if int(self.pipeline_metrics['uptime_seconds']) % 30 == 0:
                    await self._log_health_status()
                
                # Send CloudWatch metrics every minute
                if int(self.pipeline_metrics['uptime_seconds']) % 60 == 0:
                    await self._send_cloudwatch_metrics()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _log_health_status(self):
        """Log current health status"""
        try:
            # Get component metrics
            simulator_metrics = self.simulator.get_metrics() if self.simulator else {}
            processor_metrics = self.processor.get_metrics() if self.processor else {}
            validator_metrics = self.validator.get_metrics() if self.validator else {}
            
            logger.info("=" * 40)
            logger.info("PIPELINE HEALTH STATUS")
            logger.info("=" * 40)
            logger.info(f"Uptime: {self.pipeline_metrics['uptime_seconds']:.0f}s")
            logger.info(f"Throughput: {self.pipeline_metrics['avg_throughput']:.1f} readings/sec")
            logger.info(f"Total Processed: {self.pipeline_metrics['total_readings_processed']}")
            logger.info(f"Alerts: {self.pipeline_metrics['alerts_generated']}")
            logger.info(f"Errors: {self.pipeline_metrics['errors_encountered']}")
            
            # Component health
            logger.info("Component Status:")
            for component, status in self.components_health.items():
                status_emoji = {'running': '✅', 'stopped': '⏹️', 'failed': '❌', 'initialized': '🔄'}
                emoji = status_emoji.get(status, '❓')
                logger.info(f"  {component}: {emoji} {status}")
            
            logger.info("=" * 40)
            
        except Exception as e:
            logger.error(f"Health status logging error: {e}")
    
    async def _send_cloudwatch_metrics(self):
        """Send comprehensive metrics to CloudWatch"""
        if not self.cloudwatch:
            return
        
        try:
            # Prepare comprehensive metrics
            cloudwatch_metrics = {
                'total_records_processed': self.pipeline_metrics['total_readings_processed'],
                'processing_time_ms': self.pipeline_metrics['avg_throughput'] * 1000 if self.pipeline_metrics['avg_throughput'] > 0 else 0,
                'error_count': self.pipeline_metrics['errors_encountered'],
                'anomaly_count': self.pipeline_metrics['alerts_generated'],
                'success': self.running and all(
                    status in ['running', 'initialized'] 
                    for status in self.components_health.values()
                ),
                'uptime_seconds': self.pipeline_metrics['uptime_seconds']
            }
            
            # Send to CloudWatch
            success = self.cloudwatch.send_pipeline_metrics(cloudwatch_metrics)
            if success:
                logger.debug("Sent pipeline metrics to CloudWatch")
            else:
                logger.warning("Failed to send metrics to CloudWatch")
                
        except Exception as e:
            logger.error(f"CloudWatch metrics error: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown the pipeline"""
        if not self.running:
            return
        
        logger.info("Initiating graceful pipeline shutdown...")
        self.running = False
        
        try:
            # Stop components in reverse order
            if self.simulator:
                await self.simulator.stop()
                self.components_health['simulator'] = 'stopped'
                logger.info("✅ Simulator stopped")
            
            if self.validator:
                await self.validator.stop_validation_service()
                self.components_health['validator'] = 'stopped'
                logger.info("✅ Validator stopped")
            
            if self.processor:
                await self.processor.stop_processing()
                self.components_health['processor'] = 'stopped'
                logger.info("✅ Processor stopped")
            
            # Final metrics to CloudWatch
            if self.cloudwatch:
                await self._send_cloudwatch_metrics()
                self.components_health['cloudwatch'] = 'stopped'
                logger.info("✅ Final metrics sent to CloudWatch")
            
            # Log final summary
            self._log_final_summary()
            
            logger.info("🎯 Pipeline shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
        
        self.shutdown_event.set()
    
    def _log_final_summary(self):
        """Log final pipeline execution summary"""
        uptime = self.pipeline_metrics['uptime_seconds']
        
        logger.info("=" * 60)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Runtime: {uptime:.1f} seconds ({uptime/60:.1f} minutes)")
        logger.info(f"Readings Processed: {self.pipeline_metrics['total_readings_processed']:,}")
        logger.info(f"Quality Validations: {self.pipeline_metrics['quality_validations']:,}")
        logger.info(f"Alerts Generated: {self.pipeline_metrics['alerts_generated']:,}")
        logger.info(f"Errors Encountered: {self.pipeline_metrics['errors_encountered']:,}")
        logger.info(f"Average Throughput: {self.pipeline_metrics['avg_throughput']:.2f} readings/sec")
        
        # Calculate success rate
        if self.pipeline_metrics['total_readings_processed'] > 0:
            success_rate = (
                (self.pipeline_metrics['total_readings_processed'] - self.pipeline_metrics['errors_encountered']) /
                self.pipeline_metrics['total_readings_processed']
            ) * 100
            logger.info(f"Success Rate: {success_rate:.1f}%")
        
        logger.info("=" * 60)

# =====================================================
# Configuration Profiles
# =====================================================

PIPELINE_CONFIGS = {
    'development': {
        'num_engines': 2,           # Reduced from 3
        'frequency_hz': 1.0,        # Reduced from 2.0
        'validation_window_size': 20, # Reduced from 50
        'validation_interval': 1.0,  # Increased from 0.5
        'enable_cloudwatch': False
    },
    'testing': {
        'num_engines': 2,
        'frequency_hz': 5.0,
        'validation_window_size': 20,
        'validation_interval': 0.2,
        'enable_cloudwatch': False
    },
    'production': {
        'num_engines': 10,
        'frequency_hz': 1.0,
        'validation_window_size': 200,
        'validation_interval': 1.0,
        'enable_cloudwatch': True
    },
    'demo': {
        'num_engines': 2,
        'frequency_hz': 3.0,
        'validation_window_size': 30,
        'validation_interval': 0.3,
        'enable_cloudwatch': False
    }
}

# =====================================================
# Main Function
# =====================================================

async def main():
    """Main function for continuous pipeline"""
    parser = argparse.ArgumentParser(description='Continuous Data Pipeline')
    parser.add_argument('--profile', choices=PIPELINE_CONFIGS.keys(), 
                       default='development', help='Pipeline configuration profile')
    parser.add_argument('--duration', type=int, help='Duration in seconds (optional)')
    parser.add_argument('--engines', type=int, help='Number of engines to simulate')
    parser.add_argument('--frequency', type=float, help='Reading frequency in Hz')
    parser.add_argument('--cloudwatch', action='store_true', help='Enable CloudWatch monitoring')
    
    args = parser.parse_args()
    
    # Get configuration
    config = PIPELINE_CONFIGS[args.profile].copy()
    
    # Override with command line arguments
    if args.engines:
        config['num_engines'] = args.engines
    if args.frequency:
        config['frequency_hz'] = args.frequency
    if args.cloudwatch:
        config['enable_cloudwatch'] = True
    
    logger.info(f"Starting continuous pipeline with profile: {args.profile}")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Create and run orchestrator
    orchestrator = ContinuousPipelineOrchestrator(config)
    
    try:
        # Initialize components
        if not await orchestrator.initialize_components():
            logger.error("Component initialization failed")
            return False
        
        # Start pipeline
        await orchestrator.start_pipeline(duration_seconds=args.duration)
        
        return True
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        await orchestrator.shutdown()
        return True
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        await orchestrator.shutdown()
        return False

def cli_main():
    """CLI entry point"""
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    cli_main()