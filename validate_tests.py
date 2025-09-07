#!/usr/bin/env python3
"""
Simple test validation script to check test suite functionality
"""
import sys
import os
import traceback

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    
    try:
        # Test pytest imports
        import pytest
        print("✓ pytest imported successfully")
        
        # Test basic app imports
        from app.config import settings
        print("✓ app.config imported successfully")
        
        # Test service imports
        from app.services.fraud_detection import FraudDetector
        print("✓ FraudDetector imported successfully")
        
        from app.services.predictive_analytics import PredictiveAnalyzer
        print("✓ PredictiveAnalyzer imported successfully")
        
        # Test model imports
        from app.models.base import FraudAlert, FraudType
        print("✓ Base models imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        # Test FraudDetector instantiation
        from app.services.fraud_detection import FraudDetector
        detector = FraudDetector()
        print("✓ FraudDetector instantiated successfully")
        
        # Test PredictiveAnalyzer instantiation
        from app.services.predictive_analytics import PredictiveAnalyzer
        analyzer = PredictiveAnalyzer()
        print("✓ PredictiveAnalyzer instantiated successfully")
        
        # Test model creation
        from app.models.base import FraudAlert, FraudType
        alert = FraudAlert(
            type=FraudType.DUPLICATE_INVOICE,
            message="Test alert",
            confidence_score=0.9,
            business_id="test_business"
        )
        print("✓ FraudAlert model created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality error: {e}")
        traceback.print_exc()
        return False


def test_simple_pytest():
    """Test simple pytest execution"""
    print("\nTesting simple pytest execution...")
    
    try:
        import subprocess
        
        # Create a simple test file
        simple_test = '''
import pytest

def test_simple_addition():
    """Simple test to verify pytest is working"""
    assert 1 + 1 == 2

def test_simple_string():
    """Simple string test"""
    assert "hello" + " world" == "hello world"
'''
        
        with open('simple_test.py', 'w') as f:
            f.write(simple_test)
        
        # Run the simple test
        result = subprocess.run(['python', '-m', 'pytest', 'simple_test.py', '-v'], 
                              capture_output=True, text=True)
        
        # Clean up
        os.remove('simple_test.py')
        
        if result.returncode == 0:
            print("✓ Simple pytest execution successful")
            return True
        else:
            print(f"✗ Pytest execution failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Pytest test error: {e}")
        return False


def test_fraud_detection_methods():
    """Test available methods in FraudDetector"""
    print("\nTesting FraudDetector methods...")
    
    try:
        from app.services.fraud_detection import FraudDetector
        detector = FraudDetector()
        
        # List available methods
        methods = [method for method in dir(detector) if not method.startswith('_')]
        print(f"Available methods: {methods}")
        
        # Test specific methods exist
        required_methods = ['analyze_fraud', 'detect_duplicates', 'detect_payment_mismatches']
        for method in required_methods:
            if hasattr(detector, method):
                print(f"✓ {method} method exists")
            else:
                print(f"✗ {method} method missing")
        
        return True
        
    except Exception as e:
        print(f"✗ Method test error: {e}")
        return False


def main():
    """Main validation function"""
    print("="*60)
    print("AI Backend Test Suite Validation")
    print("="*60)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_fraud_detection_methods,
        test_simple_pytest
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All validation tests passed!")
        print("The test suite should be ready to use.")
        return 0
    else:
        print("✗ Some validation tests failed.")
        print("Please check the errors above and fix them before running the full test suite.")
        return 1


if __name__ == "__main__":
    sys.exit(main())