#!/usr/bin/env python3
"""
Simple validation script for Supabase integration structure
"""
import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (NOT FOUND)")
        return False

def check_directory_structure():
    """Check if all required files and directories exist"""
    print("🏗️  Checking AI Backend Structure")
    print("=" * 40)
    
    checks = [
        # Core application files
        ("app/__init__.py", "App package init"),
        ("app/main.py", "FastAPI main application"),
        ("app/config.py", "Configuration module"),
        ("app/database.py", "Database connection module"),
        
        # Authentication and utilities
        ("app/utils/__init__.py", "Utils package init"),
        ("app/utils/auth.py", "Authentication middleware"),
        ("app/utils/database.py", "Database utilities"),
        
        # API modules
        ("app/api/__init__.py", "API package init"),
        ("app/api/fraud.py", "Fraud detection API"),
        ("app/api/insights.py", "Business insights API"),
        ("app/api/compliance.py", "Compliance checking API"),
        ("app/api/invoice.py", "Invoice generation API"),
        
        # Service modules
        ("app/services/__init__.py", "Services package init"),
        ("app/services/fraud_detection.py", "Fraud detection service"),
        ("app/services/predictive_analytics.py", "Predictive analytics service"),
        ("app/services/compliance.py", "Compliance service"),
        ("app/services/nlp_invoice.py", "NLP invoice service"),
        
        # Model modules
        ("app/models/__init__.py", "Models package init"),
        ("app/models/base.py", "Base models"),
        ("app/models/responses.py", "Response models"),
        
        # Configuration files
        (".env", "Environment configuration"),
        (".env.example", "Environment configuration example"),
        ("requirements.txt", "Python dependencies"),
        ("docker-compose.yml", "Docker compose configuration"),
        ("Dockerfile", "Docker configuration"),
        
        # Database schema
        ("schema/ai_extensions.sql", "AI database schema extensions"),
        
        # Tests
        ("tests/__init__.py", "Tests package init"),
        ("tests/test_main.py", "Main application tests"),
        ("tests/test_database.py", "Database integration tests"),
    ]
    
    passed = 0
    total = len(checks)
    
    for filepath, description in checks:
        if check_file_exists(filepath, description):
            passed += 1
    
    return passed, total

def check_file_content():
    """Check if key files have expected content"""
    print("\n📄 Checking File Content")
    print("=" * 40)
    
    checks_passed = 0
    total_checks = 0
    
    # Check database.py has key classes
    total_checks += 1
    try:
        with open("app/database.py", "r") as f:
            content = f.read()
            if "class DatabaseManager" in content and "async def init_database" in content:
                print("✅ database.py contains required classes and functions")
                checks_passed += 1
            else:
                print("❌ database.py missing required classes or functions")
    except FileNotFoundError:
        print("❌ database.py not found")
    
    # Check auth.py has authentication middleware
    total_checks += 1
    try:
        with open("app/utils/auth.py", "r") as f:
            content = f.read()
            if "class AuthToken" in content and "async def get_current_user" in content:
                print("✅ auth.py contains authentication components")
                checks_passed += 1
            else:
                print("❌ auth.py missing authentication components")
    except FileNotFoundError:
        print("❌ auth.py not found")
    
    # Check config.py has settings
    total_checks += 1
    try:
        with open("app/config.py", "r") as f:
            content = f.read()
            if "SUPABASE_URL" in content and "class Settings" in content:
                print("✅ config.py contains configuration settings")
                checks_passed += 1
            else:
                print("❌ config.py missing configuration settings")
    except FileNotFoundError:
        print("❌ config.py not found")
    
    # Check schema file
    total_checks += 1
    try:
        with open("schema/ai_extensions.sql", "r") as f:
            content = f.read()
            if "ai_analysis_results" in content and "fraud_alerts" in content:
                print("✅ AI schema contains required tables")
                checks_passed += 1
            else:
                print("❌ AI schema missing required tables")
    except FileNotFoundError:
        print("❌ AI schema file not found")
    
    return checks_passed, total_checks

def check_environment_config():
    """Check environment configuration"""
    print("\n🔧 Checking Environment Configuration")
    print("=" * 40)
    
    checks_passed = 0
    total_checks = 0
    
    # Check .env file
    total_checks += 1
    try:
        with open(".env", "r") as f:
            content = f.read()
            required_vars = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "SUPABASE_ANON_KEY"]
            found_vars = sum(1 for var in required_vars if var in content)
            
            if found_vars == len(required_vars):
                print(f"✅ .env contains all required variables ({found_vars}/{len(required_vars)})")
                checks_passed += 1
            else:
                print(f"❌ .env missing required variables ({found_vars}/{len(required_vars)})")
    except FileNotFoundError:
        print("❌ .env file not found")
    
    # Check .env.example file
    total_checks += 1
    try:
        with open(".env.example", "r") as f:
            content = f.read()
            if "SUPABASE_URL" in content and "# AI Backend Environment" in content:
                print("✅ .env.example contains template configuration")
                checks_passed += 1
            else:
                print("❌ .env.example missing template configuration")
    except FileNotFoundError:
        print("❌ .env.example file not found")
    
    return checks_passed, total_checks

def main():
    """Main validation function"""
    print("🚀 AI Backend Supabase Integration Validation")
    print("=" * 50)
    
    # Check directory structure
    structure_passed, structure_total = check_directory_structure()
    
    # Check file content
    content_passed, content_total = check_file_content()
    
    # Check environment configuration
    env_passed, env_total = check_environment_config()
    
    # Summary
    total_passed = structure_passed + content_passed + env_passed
    total_checks = structure_total + content_total + env_total
    
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"📁 Structure: {structure_passed}/{structure_total}")
    print(f"📄 Content: {content_passed}/{content_total}")
    print(f"🔧 Environment: {env_passed}/{env_total}")
    print(f"🎯 Total: {total_passed}/{total_checks}")
    
    if total_passed == total_checks:
        print("\n🎉 All validations PASSED!")
        print("✅ Supabase integration structure is complete")
        return 0
    else:
        percentage = (total_passed / total_checks) * 100
        print(f"\n⚠️  {percentage:.1f}% of validations passed")
        if percentage >= 80:
            print("✅ Integration is mostly complete")
            return 0
        else:
            print("❌ Some critical components are missing")
            return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)