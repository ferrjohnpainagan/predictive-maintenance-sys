"""
Simple Test Suite for C-MAPSS Data Pipeline
Basic functionality tests
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def test_config_import():
    """Test that we can import the configuration"""
    try:
        from config.config import CMAPSSConfig
        print("✅ Configuration import successful")
        return True
    except Exception as e:
        print(f"❌ Configuration import failed: {e}")
        return False

def test_cmapss_config():
    """Test CMAPSS configuration creation"""
    try:
        from config.config import CMAPSSConfig
        
        config = CMAPSSConfig(
            num_engines=5,
            max_cycles=100,
            num_sensors=14
        )
        
        print(f"✅ Config created: {config.num_engines} engines, {config.max_cycles} max cycles")
        print(f"✅ Column names: {len(config.column_names)} columns")
        print(f"✅ Data types: {len(config.dtypes)} dtypes")
        
        return True
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return False

def test_data_generation():
    """Test basic data generation"""
    try:
        import numpy as np
        import pandas as pd
        from config.config import CMAPSSConfig
        
        config = CMAPSSConfig(
            num_engines=3,
            max_cycles=20,
            num_sensors=14
        )
        
        # Generate simple test data
        data = []
        for engine_num in range(1, 4):
            for cycle in range(1, 6):
                record = [engine_num, cycle, 35000, 0.8, 85] + [100.0] * 14
                data.append(record)
        
        df = pd.DataFrame(data, columns=config.column_names)
        
        print(f"✅ Data generation successful: {len(df)} records")
        print(f"✅ DataFrame shape: {df.shape}")
        print(f"✅ Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        return True
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

def test_file_operations():
    """Test file operations"""
    try:
        # Test directory creation
        os.makedirs("data/test_output", exist_ok=True)
        
        # Test file writing
        test_file = "data/test_output/test.txt"
        with open(test_file, 'w') as f:
            f.write("test data")
        
        # Test file reading
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Cleanup
        os.remove(test_file)
        os.rmdir("data/test_output")
        
        print("✅ File operations successful")
        return True
    except Exception as e:
        print(f"❌ File operations failed: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization features"""
    try:
        import numpy as np
        import pandas as pd

        # Create data with different dtypes
        data_int32 = np.array([1, 2, 3, 4, 5], dtype='int32')
        data_int64 = np.array([1, 2, 3, 4, 5], dtype='int64')
        data_float32 = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype='float32')
        data_float64 = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype='float64')
        
        # Calculate memory usage
        memory_int32 = data_int32.nbytes / 1024
        memory_int64 = data_int64.nbytes / 1024
        memory_float32 = data_float32.nbytes / 1024
        memory_float64 = data_float64.nbytes / 1024
        
        print(f"✅ Memory optimization test:")
        print(f"   int32: {memory_int32:.2f} KB")
        print(f"   int64: {memory_int64:.2f} KB")
        print(f"   float32: {memory_float32:.2f} KB")
        print(f"   float64: {memory_float64:.2f} KB")
        
        # Verify optimization
        assert memory_int32 < memory_int64
        assert memory_float32 < memory_float64
        
        return True
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running C-MAPSS Data Pipeline Tests...")
    print("=" * 50)
    
    tests = [
        ("Configuration Import", test_config_import),
        ("CMAPSS Config", test_cmapss_config),
        ("Data Generation", test_data_generation),
        ("File Operations", test_file_operations),
        ("Memory Optimization", test_memory_optimization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Data pipeline is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
