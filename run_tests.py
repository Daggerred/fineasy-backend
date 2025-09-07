#!/usr/bin/env python3
"""
Comprehensive test runner for AI backend services
"""
import os
import sys
import subprocess
import argparse
import time
from pathlib import Path


def run_command(command, description):
    """Run a command and return the result"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Duration: {end_time - start_time:.2f} seconds")
    print(f"Return code: {result.returncode}")
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode == 0


def run_unit_tests():
    """Run unit tests"""
    command = "python -m pytest tests/ -m 'unit and not slow' --cov=app --cov-report=term-missing"
    return run_command(command, "Unit Tests")


def run_integration_tests():
    """Run integration tests"""
    command = "python -m pytest tests/ -m 'integration and not slow' --cov=app --cov-report=term-missing"
    return run_command(command, "Integration Tests")


def run_performance_tests():
    """Run performance tests"""
    command = "python -m pytest tests/ -m 'performance' --cov=app --cov-report=term-missing"
    return run_command(command, "Performance Tests")


def run_accuracy_tests():
    """Run accuracy validation tests"""
    command = "python -m pytest tests/ -m 'accuracy' --cov=app --cov-report=term-missing"
    return run_command(command, "Accuracy Validation Tests")


def run_smoke_tests():
    """Run smoke tests for quick validation"""
    command = "python -m pytest tests/ -m 'smoke' --maxfail=1"
    return run_command(command, "Smoke Tests")


def run_all_tests():
    """Run all tests"""
    command = "python -m pytest tests/ --cov=app --cov-report=term-missing --cov-report=html:htmlcov"
    return run_command(command, "All Tests")


def run_specific_test_file(test_file):
    """Run a specific test file"""
    command = f"python -m pytest {test_file} -v --cov=app --cov-report=term-missing"
    return run_command(command, f"Specific Test File: {test_file}")


def run_tests_with_coverage():
    """Run tests with detailed coverage report"""
    command = "python -m pytest tests/ --cov=app --cov-report=term-missing --cov-report=html:htmlcov --cov-report=xml:coverage.xml"
    return run_command(command, "Tests with Coverage Report")


def run_parallel_tests():
    """Run tests in parallel"""
    try:
        import pytest_xdist
        command = "python -m pytest tests/ -n auto --cov=app --cov-report=term-missing"
        return run_command(command, "Parallel Tests")
    except ImportError:
        print("pytest-xdist not installed. Running tests sequentially.")
        return run_all_tests()


def lint_code():
    """Run code linting"""
    commands = [
        ("python -m flake8 app/ --max-line-length=120 --ignore=E203,W503", "Flake8 Linting"),
        ("python -m black --check app/", "Black Code Formatting Check"),
        ("python -m mypy app/ --ignore-missing-imports", "MyPy Type Checking")
    ]
    
    all_passed = True
    for command, description in commands:
        if not run_command(command, description):
            all_passed = False
    
    return all_passed


def generate_test_report():
    """Generate comprehensive test report"""
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE TEST REPORT")
    print("="*80)
    
    # Run tests with JUnit XML output for CI/CD
    command = "python -m pytest tests/ --junitxml=test-results.xml --cov=app --cov-report=xml:coverage.xml --cov-report=html:htmlcov"
    success = run_command(command, "Test Report Generation")
    
    if success:
        print("\nTest report generated successfully!")
        print("- JUnit XML: test-results.xml")
        print("- Coverage XML: coverage.xml")
        print("- Coverage HTML: htmlcov/index.html")
    
    return success


def validate_test_environment():
    """Validate test environment setup"""
    print("\n" + "="*60)
    print("VALIDATING TEST ENVIRONMENT")
    print("="*60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        return False
    
    # Check required packages
    required_packages = [
        'pytest', 'pytest-asyncio', 'pytest-cov', 'fastapi', 'uvicorn',
        'pandas', 'numpy', 'scikit-learn', 'spacy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nERROR: Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    # Check test data directory
    test_data_dir = Path("tests/data")
    if not test_data_dir.exists():
        print(f"Creating test data directory: {test_data_dir}")
        test_data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nTest environment validation completed successfully!")
    return True


def cleanup_test_artifacts():
    """Clean up test artifacts"""
    print("\n" + "="*60)
    print("CLEANING UP TEST ARTIFACTS")
    print("="*60)
    
    artifacts = [
        ".coverage",
        "coverage.xml",
        "test-results.xml",
        "htmlcov/",
        ".pytest_cache/",
        "__pycache__/",
        "*.pyc"
    ]
    
    for artifact in artifacts:
        if os.path.exists(artifact):
            if os.path.isdir(artifact):
                import shutil
                shutil.rmtree(artifact)
                print(f"Removed directory: {artifact}")
            else:
                os.remove(artifact)
                print(f"Removed file: {artifact}")
    
    # Remove __pycache__ directories recursively
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_path = os.path.join(root, dir_name)
                import shutil
                shutil.rmtree(pycache_path)
                print(f"Removed: {pycache_path}")
    
    print("Cleanup completed!")


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="AI Backend Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--accuracy", action="store_true", help="Run accuracy tests only")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage report")
    parser.add_argument("--lint", action="store_true", help="Run code linting")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive test report")
    parser.add_argument("--validate", action="store_true", help="Validate test environment")
    parser.add_argument("--cleanup", action="store_true", help="Clean up test artifacts")
    parser.add_argument("--file", type=str, help="Run specific test file")
    parser.add_argument("--ci", action="store_true", help="Run in CI mode (all tests + linting + report)")
    
    args = parser.parse_args()
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    success = True
    
    # Validate environment first
    if args.validate or args.ci:
        if not validate_test_environment():
            sys.exit(1)
    
    # Clean up if requested
    if args.cleanup:
        cleanup_test_artifacts()
        return
    
    # Run linting if requested
    if args.lint or args.ci:
        if not lint_code():
            success = False
    
    # Run specific test categories
    if args.unit:
        success &= run_unit_tests()
    elif args.integration:
        success &= run_integration_tests()
    elif args.performance:
        success &= run_performance_tests()
    elif args.accuracy:
        success &= run_accuracy_tests()
    elif args.smoke:
        success &= run_smoke_tests()
    elif args.parallel:
        success &= run_parallel_tests()
    elif args.coverage:
        success &= run_tests_with_coverage()
    elif args.file:
        success &= run_specific_test_file(args.file)
    elif args.report or args.ci:
        success &= generate_test_report()
    elif args.all or args.ci:
        success &= run_all_tests()
    else:
        # Default: run unit and integration tests
        print("No specific test category specified. Running unit and integration tests.")
        success &= run_unit_tests()
        success &= run_integration_tests()
    
    # Generate report for CI mode
    if args.ci and success:
        success &= generate_test_report()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST EXECUTION SUMMARY")
    print("="*80)
    
    if success:
        print("✓ All tests passed successfully!")
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()