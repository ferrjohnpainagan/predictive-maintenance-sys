#!/usr/bin/env python3
"""
Integration Test for Continuous Pipeline
Tests the complete streaming pipeline integration
"""

import asyncio
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.continuous_pipeline import ContinuousPipelineOrchestrator, PIPELINE_CONFIGS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_pipeline_initialization():
    """Test pipeline component initialization"""
    logger.info("Testing pipeline initialization...")
    
    config = PIPELINE_CONFIGS['testing'].copy()
    orchestrator = ContinuousPipelineOrchestrator(config)
    
    try:
        # Test initialization
        success = await orchestrator.initialize_components()
        assert success, "Pipeline initialization failed"
        
        # Check component health
        assert orchestrator.components_health['simulator'] == 'initialized'
        assert orchestrator.components_health['processor'] == 'initialized' 
        assert orchestrator.components_health['validator'] == 'initialized'
        
        logger.info("✅ Pipeline initialization test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline initialization test failed: {e}")
        return False
    finally:
        await orchestrator.shutdown()

async def test_pipeline_execution():
    """Test short pipeline execution"""
    logger.info("Testing pipeline execution...")
    
    config = PIPELINE_CONFIGS['testing'].copy()
    config['num_engines'] = 2
    config['frequency_hz'] = 2.0
    
    orchestrator = ContinuousPipelineOrchestrator(config)
    
    try:
        # Initialize components
        success = await orchestrator.initialize_components()
        assert success, "Pipeline initialization failed"
        
        # Run for 10 seconds
        await orchestrator.start_pipeline(duration_seconds=10)
        
        # Check metrics
        metrics = orchestrator.pipeline_metrics
        assert metrics['total_readings_processed'] > 0, "No readings processed"
        assert metrics['quality_validations'] > 0, "No validations performed"
        
        logger.info(f"✅ Pipeline execution test passed - Processed {metrics['total_readings_processed']} readings")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution test failed: {e}")
        return False

async def test_component_integration():
    """Test component integration and data flow"""
    logger.info("Testing component integration...")
    
    from scripts.sensor_simulator import SensorSimulator
    from scripts.streaming_processor import StreamingProcessor
    from scripts.realtime_validator import RealtimeValidator
    
    try:
        # Create components
        simulator = SensorSimulator(num_engines=1, frequency_hz=1.0)
        processor = StreamingProcessor()
        validator = RealtimeValidator(window_size=10)
        
        # Track processed data
        readings_processed = []
        validations_completed = []
        
        # Set up data flow
        async def process_and_validate(reading):
            # Process
            result = await processor.process_reading(reading)
            readings_processed.append(result)
            
            # Validate
            validation = await validator.validate_reading(reading)
            validations_completed.append(validation)
        
        simulator.register_callback('on_reading', process_and_validate)
        
        # Start components
        await processor.start_processing()
        await validator.start_validation_service()
        
        # Run for 5 seconds
        await simulator.start(duration_seconds=5)
        
        # Stop components
        await processor.stop_processing()
        await validator.stop_validation_service()
        
        # Verify data flow
        assert len(readings_processed) > 0, "No readings processed"
        assert len(validations_completed) > 0, "No validations completed"
        assert len(readings_processed) == len(validations_completed), "Processing/validation count mismatch"
        
        logger.info(f"✅ Component integration test passed - {len(readings_processed)} readings processed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Component integration test failed: {e}")
        return False

async def test_error_handling():
    """Test pipeline error handling and recovery"""
    logger.info("Testing error handling...")
    
    config = PIPELINE_CONFIGS['testing'].copy()
    orchestrator = ContinuousPipelineOrchestrator(config)
    
    try:
        # Initialize components
        success = await orchestrator.initialize_components()
        assert success, "Pipeline initialization failed"
        
        # Test with invalid configuration
        orchestrator.simulator.engines[1].inject_anomaly('failure')
        
        # Run briefly
        await orchestrator.start_pipeline(duration_seconds=3)
        
        # Should have some errors but continue running
        metrics = orchestrator.pipeline_metrics
        logger.info(f"Processed {metrics['total_readings_processed']} readings with {metrics['errors_encountered']} errors")
        
        logger.info("✅ Error handling test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all continuous pipeline tests"""
    print("\n" + "=" * 60)
    print("CONTINUOUS PIPELINE INTEGRATION TESTS")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}\n")
    
    tests = [
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Pipeline Execution", test_pipeline_execution),
        ("Component Integration", test_component_integration),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    
    async def run_tests():
        for test_name, test_func in tests:
            print(f"\n📝 Running: {test_name}")
            print("-" * 40)
            try:
                success = await test_func()
                results.append((test_name, success))
            except Exception as e:
                logger.error(f"Test {test_name} crashed: {e}")
                results.append((test_name, False))
            print()
    
    # Run all tests
    asyncio.run(run_tests())
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:30} {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All continuous pipeline tests passed!")
        print("✅ Phase 3 implementation verified!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("❌ Please check component configuration")
    
    print("=" * 60 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)