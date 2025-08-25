#!/usr/bin/env python3
"""
Performance Testing Script
Tests the system performance within free tier constraints
"""

import os
import sys
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.performance_tester import PerformanceTester

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run performance tests"""
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