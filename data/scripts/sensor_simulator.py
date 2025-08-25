"""
Real-time Sensor Data Simulator
Generates realistic aircraft engine sensor data in real-time
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import random

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import get_config
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
# Data Models
# =====================================================

@dataclass
class SensorReading:
    """Single sensor reading data model"""
    engine_id: str
    unit_number: int
    timestamp: datetime
    time_cycles: int
    
    # Operational settings
    altitude: float
    mach_number: float
    tra: float  # Throttle resolver angle
    
    # Sensor readings
    s2: float   # T24 - Total temperature at LPC outlet
    s3: float   # T30 - Total temperature at HPC outlet
    s4: float   # T50 - Total temperature at LPT outlet
    s7: float   # P30 - Total pressure at HPC outlet
    s8: float   # Nf - Physical fan speed
    s9: float   # Nc - Physical core speed
    s11: float  # Ps30 - Static pressure at HPC outlet
    s12: float  # phi - Ratio of fuel flow to Ps30
    s13: float  # NRf - Corrected fan speed
    s14: float  # NRc - Corrected core speed
    s15: float  # BPR - Bypass Ratio
    s17: float  # htBleed - HPT coolant bleed
    s20: float  # W31 - LPT coolant bleed
    s21: float  # W32 - HPT coolant bleed
    
    # Calculated fields
    rul: int
    health_score: float
    anomaly_score: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

# =====================================================
# Engine Simulator Class
# =====================================================

class EngineSimulator:
    """Simulates a single aircraft engine with degradation"""
    
    def __init__(self, unit_number: int, initial_health: float = 100.0):
        self.unit_number = unit_number
        self.engine_id = f"engine_{unit_number}"
        self.time_cycles = 0
        self.health = initial_health
        self.max_rul = 125
        self.current_rul = self.max_rul
        
        # Degradation parameters
        self.degradation_rate = random.uniform(0.1, 0.3)  # Per cycle
        self.noise_level = random.uniform(0.02, 0.05)
        
        # Operational profile
        self.altitude_range = (30000, 42000)
        self.mach_range = (0.7, 0.85)
        self.tra_range = (80, 95)
        
        # Sensor baselines
        self.sensor_baselines = {
            's2': 640, 's3': 1600, 's4': 1400, 
            's7': 3.0, 's8': 9500, 's9': 8750,
            's11': 3.0, 's12': 0.5, 's13': 9500, 
            's14': 8750, 's15': 5.5, 's17': 0.1, 
            's20': 0.1, 's21': 0.1
        }
        
        # Failure modes
        self.failure_modes = {
            'fan_degradation': False,
            'compressor_fault': False,
            'turbine_wear': False,
            'sensor_drift': False
        }
        
        # Initialize failure mode randomly
        if random.random() < 0.3:  # 30% chance of failure mode
            failure_type = random.choice(list(self.failure_modes.keys()))
            self.failure_modes[failure_type] = True
            logger.info(f"Engine {unit_number} initialized with {failure_type}")
    
    def update_health(self):
        """Update engine health and RUL"""
        # Apply degradation
        self.health -= self.degradation_rate
        self.health = max(0, self.health)
        
        # Update RUL
        self.current_rul = int(self.health / 100 * self.max_rul)
        
        # Accelerate degradation if failure mode active
        if any(self.failure_modes.values()):
            self.degradation_rate *= 1.01  # Accelerating degradation
        
        # Update time cycles
        self.time_cycles += 1
    
    def generate_sensor_reading(self) -> SensorReading:
        """Generate a realistic sensor reading"""
        # Update health first
        self.update_health()
        
        # Generate operational settings
        altitude = random.uniform(*self.altitude_range)
        mach_number = random.uniform(*self.mach_range)
        tra = random.uniform(*self.tra_range)
        
        # Calculate degradation factor
        degradation_factor = 1 + ((100 - self.health) / 100) * 0.2
        
        # Generate sensor readings with degradation and noise
        sensor_readings = {}
        for sensor, baseline in self.sensor_baselines.items():
            # Apply degradation
            degraded_value = baseline * degradation_factor
            
            # Apply failure mode effects
            if self.failure_modes['fan_degradation'] and sensor in ['s8', 's13']:
                degraded_value *= 1.1  # Fan speed increases
            elif self.failure_modes['compressor_fault'] and sensor in ['s3', 's7', 's11']:
                degraded_value *= 0.9  # Pressure/temperature drops
            elif self.failure_modes['turbine_wear'] and sensor in ['s4', 's17', 's20', 's21']:
                degraded_value *= 1.15  # Temperature increases
            elif self.failure_modes['sensor_drift'] and random.random() < 0.1:
                degraded_value *= random.uniform(0.95, 1.05)  # Random drift
            
            # Add noise
            noise = np.random.normal(0, degraded_value * self.noise_level)
            sensor_readings[sensor] = max(0, degraded_value + noise)
        
        # Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(sensor_readings)
        
        # Create sensor reading
        reading = SensorReading(
            engine_id=self.engine_id,
            unit_number=self.unit_number,
            timestamp=datetime.utcnow(),
            time_cycles=self.time_cycles,
            altitude=altitude,
            mach_number=mach_number,
            tra=tra,
            **sensor_readings,
            rul=self.current_rul,
            health_score=self.health,
            anomaly_score=anomaly_score
        )
        
        return reading
    
    def _calculate_anomaly_score(self, sensor_readings: Dict[str, float]) -> float:
        """Calculate anomaly score based on sensor deviations"""
        deviations = []
        for sensor, value in sensor_readings.items():
            baseline = self.sensor_baselines.get(sensor, value)
            if baseline > 0:
                deviation = abs(value - baseline) / baseline
                deviations.append(deviation)
        
        # Average deviation as anomaly score (0-100 scale)
        avg_deviation = np.mean(deviations) if deviations else 0
        anomaly_score = min(100, avg_deviation * 100)
        
        # Increase score if failure mode active
        if any(self.failure_modes.values()):
            anomaly_score = min(100, anomaly_score * 1.5)
        
        return anomaly_score
    
    def inject_anomaly(self, anomaly_type: str = 'spike'):
        """Inject an anomaly into the sensor readings"""
        if anomaly_type == 'spike':
            # Temporary spike in readings
            for sensor in ['s2', 's3', 's4']:
                self.sensor_baselines[sensor] *= 1.2
        elif anomaly_type == 'drift':
            # Gradual drift in sensor
            self.failure_modes['sensor_drift'] = True
        elif anomaly_type == 'failure':
            # Simulate impending failure
            self.degradation_rate *= 2
            self.health = min(20, self.health)

# =====================================================
# Sensor Simulator Class
# =====================================================

class SensorSimulator:
    """Real-time sensor data simulator for multiple engines"""
    
    def __init__(self, num_engines: int = 5, frequency_hz: float = 1.0):
        self.config = get_config()
        self.num_engines = num_engines
        self.frequency_hz = frequency_hz
        self.interval_seconds = 1.0 / frequency_hz
        
        # Initialize engines
        self.engines = {}
        for i in range(1, num_engines + 1):
            initial_health = random.uniform(60, 100)  # Varying initial conditions
            self.engines[i] = EngineSimulator(i, initial_health)
        
        # Data buffer
        self.data_buffer = []
        self.buffer_size = 100  # Batch size for database insertion
        
        # Metrics
        self.metrics = {
            'readings_generated': 0,
            'readings_sent': 0,
            'anomalies_detected': 0,
            'failures_predicted': 0
        }
        
        # Callbacks
        self.callbacks = {
            'on_reading': [],
            'on_anomaly': [],
            'on_batch_ready': []
        }
        
        # Database and validator
        self.db_connector = None
        self.validator = DataQualityValidator()
        
        self.running = False
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for an event"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    async def start(self, duration_seconds: Optional[int] = None):
        """Start the sensor simulation"""
        logger.info(f"Starting sensor simulation for {self.num_engines} engines at {self.frequency_hz} Hz")
        
        # Initialize database connection
        try:
            self.db_connector = SupabaseConnector(self.config.database)
            if not self.db_connector.test_connection():
                logger.warning("Database connection failed, running in offline mode")
                self.db_connector = None
        except Exception as e:
            logger.warning(f"Could not connect to database: {e}")
            self.db_connector = None
        
        self.running = True
        start_time = time.time()
        
        try:
            while self.running:
                # Check duration limit
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    break
                
                # Generate readings for all engines
                await self._generate_readings()
                
                # Sleep until next reading
                await asyncio.sleep(self.interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        finally:
            await self.stop()
    
    async def _generate_readings(self):
        """Generate readings for all engines"""
        readings = []
        
        for engine in self.engines.values():
            reading = engine.generate_sensor_reading()
            readings.append(reading)
            
            # Update metrics
            self.metrics['readings_generated'] += 1
            
            # Check for anomalies
            if reading.anomaly_score > 50:
                self.metrics['anomalies_detected'] += 1
                await self._handle_anomaly(reading)
            
            # Check for low RUL
            if reading.rul < 20:
                self.metrics['failures_predicted'] += 1
                await self._handle_low_rul(reading)
            
            # Trigger callbacks
            for callback in self.callbacks['on_reading']:
                await self._run_callback(callback, reading)
        
        # Add to buffer
        self.data_buffer.extend(readings)
        
        # Process buffer if full
        if len(self.data_buffer) >= self.buffer_size:
            await self._process_buffer()
    
    async def _process_buffer(self):
        """Process and send buffered data"""
        if not self.data_buffer:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([r.to_dict() for r in self.data_buffer])
        
        # Validate data quality
        validation_results = self.validator.validate_dataframe(df)
        
        if validation_results['success']:
            # Send to database if connected
            if self.db_connector:
                try:
                    success, results = self.db_connector.insert_sensor_data_batch(df)
                    if success:
                        self.metrics['readings_sent'] += len(self.data_buffer)
                        logger.info(f"Sent batch of {len(self.data_buffer)} readings to database")
                    else:
                        logger.error(f"Failed to send batch: {results}")
                except Exception as e:
                    logger.error(f"Database insertion error: {e}")
            else:
                # Save to local file in offline mode
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                filename = f"data/streaming/sensor_data_{timestamp}.parquet"
                os.makedirs("data/streaming", exist_ok=True)
                df.to_parquet(filename)
                logger.info(f"Saved batch to {filename} (offline mode)")
        else:
            logger.warning(f"Data quality validation failed: {validation_results}")
        
        # Trigger batch callbacks
        for callback in self.callbacks['on_batch_ready']:
            await self._run_callback(callback, df)
        
        # Clear buffer
        self.data_buffer.clear()
    
    async def _handle_anomaly(self, reading: SensorReading):
        """Handle anomaly detection"""
        logger.warning(f"Anomaly detected - Engine {reading.unit_number}: "
                      f"Score={reading.anomaly_score:.1f}, RUL={reading.rul}")
        
        # Trigger anomaly callbacks
        for callback in self.callbacks['on_anomaly']:
            await self._run_callback(callback, reading)
    
    async def _handle_low_rul(self, reading: SensorReading):
        """Handle low RUL warning"""
        logger.warning(f"⚠️ LOW RUL WARNING - Engine {reading.unit_number}: "
                      f"RUL={reading.rul} cycles remaining")
    
    async def _run_callback(self, callback: Callable, data):
        """Run a callback safely"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
        except Exception as e:
            logger.error(f"Callback error: {e}")
    
    async def stop(self):
        """Stop the simulation"""
        self.running = False
        
        # Process remaining buffer
        if self.data_buffer:
            await self._process_buffer()
        
        # Close database connection
        if self.db_connector:
            self.db_connector.close()
        
        # Print metrics
        logger.info("Simulation stopped")
        logger.info(f"Metrics: {json.dumps(self.metrics, indent=2)}")
    
    def inject_anomaly(self, unit_number: int, anomaly_type: str = 'spike'):
        """Inject an anomaly into a specific engine"""
        if unit_number in self.engines:
            self.engines[unit_number].inject_anomaly(anomaly_type)
            logger.info(f"Injected {anomaly_type} anomaly into engine {unit_number}")
    
    def get_metrics(self) -> Dict:
        """Get current simulation metrics"""
        return {
            **self.metrics,
            'engines': {
                unit: {
                    'health': engine.health,
                    'rul': engine.current_rul,
                    'cycles': engine.time_cycles,
                    'failure_modes': engine.failure_modes
                }
                for unit, engine in self.engines.items()
            }
        }

# =====================================================
# Utility Functions
# =====================================================

async def demo_simulation():
    """Demo simulation with callbacks"""
    
    # Define callbacks
    async def on_anomaly(reading: SensorReading):
        print(f"🚨 ANOMALY: Engine {reading.unit_number} - Score: {reading.anomaly_score:.1f}")
    
    def on_batch(df: pd.DataFrame):
        print(f"📦 Batch ready: {len(df)} readings")
    
    # Create simulator
    simulator = SensorSimulator(num_engines=3, frequency_hz=2.0)
    
    # Register callbacks
    simulator.register_callback('on_anomaly', on_anomaly)
    simulator.register_callback('on_batch_ready', on_batch)
    
    # Inject some anomalies
    await asyncio.sleep(5)
    simulator.inject_anomaly(1, 'spike')
    
    await asyncio.sleep(5)
    simulator.inject_anomaly(2, 'drift')
    
    # Run for 30 seconds
    await simulator.start(duration_seconds=30)

# =====================================================
# Main Function
# =====================================================

def main():
    """Main function for sensor simulator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Sensor Data Simulator')
    parser.add_argument('--engines', type=int, default=5, help='Number of engines to simulate')
    parser.add_argument('--frequency', type=float, default=1.0, help='Reading frequency in Hz')
    parser.add_argument('--duration', type=int, help='Duration in seconds (optional)')
    parser.add_argument('--demo', action='store_true', help='Run demo mode')
    
    args = parser.parse_args()
    
    if args.demo:
        # Run demo
        asyncio.run(demo_simulation())
    else:
        # Create and run simulator
        simulator = SensorSimulator(
            num_engines=args.engines,
            frequency_hz=args.frequency
        )
        
        # Run simulation
        asyncio.run(simulator.start(duration_seconds=args.duration))

if __name__ == "__main__":
    main()