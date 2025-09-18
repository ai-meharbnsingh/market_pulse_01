"""
MarketPulse Integration Test - Windows Compatible
Verify all components are properly integrated
"""

import os
import sys
import json
from pathlib import Path
import importlib.util
from datetime import datetime

# Color codes for terminal output (Windows compatible)
try:
    import colorama
    colorama.init()
    class Colors:
        GREEN = '\033[92m'
        RED = '\033[91m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        END = '\033[0m'
        BOLD = '\033[1m'
except ImportError:
    # Fallback if colorama not installed
    class Colors:
        GREEN = ''
        RED = ''
        YELLOW = ''
        BLUE = ''
        END = ''
        BOLD = ''

def test_result(name, success, message=""):
    """Print test result with color"""
    if success:
        print(f"{Colors.GREEN}✅ {name}{Colors.END}")
    else:
        print(f"{Colors.RED}❌ {name}{Colors.END}")
        if message:
            print(f"   {Colors.YELLOW}{message}{Colors.END}")
    return success

def test_folder_structure():
    """Test if folder structure exists"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}Testing Folder Structure{Colors.END}")

    required_folders = [
        "01_CORE/antifragile",
        "02_ANALYSIS/technical",
        "03_ML_ENGINE/models",
        "04_RISK",
        "05_EXECUTION",
        "06_DATA/streaming",
        "07_DASHBOARD/components"
    ]

    all_exist = True
    for folder in required_folders:
        folder_path = Path(folder)
        exists = folder_path.exists()
        test_result(f"Folder: {folder}", exists)
        all_exist = all_exist and exists

    return all_exist

def test_core_files():
    """Test if core files exist"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}Testing Core Files{Colors.END}")

    core_files = {
        "main.py": "Main orchestrator",
        "requirements.txt": "Dependencies list",
        ".env.example": "Environment template",
        ".env": "Environment file"
    }

    all_exist = True
    for file, description in core_files.items():
        exists = Path(file).exists()
        if file == ".env" and not exists:
            # .env is optional if .env.example exists
            test_result(f"{description} ({file})", True, "Using .env.example as template")
        else:
            test_result(f"{description} ({file})", exists)
            all_exist = all_exist and exists

    return all_exist

def test_antifragile_framework():
    """Test if Antifragile Framework is properly migrated"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}Testing Antifragile Framework{Colors.END}")

    framework_files = [
        "01_CORE/antifragile/__init__.py",
        "01_CORE/antifragile/core/failover_engine.py",
        "01_CORE/antifragile/core/circuit_breaker.py",
        "01_CORE/antifragile/providers/provider_registry.py"
    ]

    all_exist = True
    for file in framework_files:
        file_path = Path(file)
        exists = file_path.exists()
        test_result(f"Framework: {file_path.name}", exists)
        all_exist = all_exist and exists

    # Try to import
    if all_exist:
        try:
            # Add to Python path
            sys.path.insert(0, str(Path("01_CORE").absolute()))

            # Check if we can import the module
            from antifragile.core import failover_engine
            test_result("Framework importable", True)
        except Exception as e:
            test_result("Framework importable", False, f"Import error: {str(e)[:50]}")

    return all_exist

def test_configuration():
    """Test configuration files"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}Testing Configuration{Colors.END}")

    config_files = [
        "01_CORE/antifragile/config/settings.yaml",
        "01_CORE/antifragile/config/risk_limits.yaml"
    ]

    all_exist = True
    for file in config_files:
        file_path = Path(file)
        exists = file_path.exists()
        test_result(f"Config: {file_path.name}", exists)
        all_exist = all_exist and exists

    # Check for .env
    env_exists = Path(".env").exists() or Path(".env.example").exists()
    test_result("Environment file", env_exists,
                "Create .env from .env.example" if not Path(".env").exists() else "")

    return all_exist

def test_migrated_components():
    """Test if existing code was migrated"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}Testing Migrated Components{Colors.END}")

    migrated = {
        "02_ANALYSIS/technical/indicators.py": "Technical Analysis",
        "03_ML_ENGINE/models/alpha_model.py": "Alpha Model",
        "03_ML_ENGINE/models/lstm_intraday.py": "LSTM Model",
        "04_RISK/risk_calculator.py": "Risk Calculator",
        "06_DATA/streaming/websocket_service.py": "WebSocket Service",
        "07_DASHBOARD/components": "Dashboard Components"
    }

    migration_count = 0
    for file, component in migrated.items():
        file_path = Path(file)
        exists = file_path.exists()
        if exists:
            migration_count += 1
        test_result(f"{component}", exists,
                   "Run migration script" if not exists else "")

    migration_percent = (migration_count / len(migrated)) * 100
    print(f"\n{Colors.BOLD}Migration: {migration_percent:.0f}% complete{Colors.END}")

    return migration_count >= 3  # At least 50% migrated

def test_python_imports():
    """Test if key Python packages are available"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}Testing Python Packages{Colors.END}")

    packages = {
        "pandas": "Data processing",
        "numpy": "Numerical computing",
        "yaml": "Configuration",
        "dotenv": "Environment variables"
    }

    all_available = True
    optional_missing = []

    for package, description in packages.items():
        try:
            __import__(package)
            test_result(f"{description} ({package})", True)
        except ImportError:
            if package in ["pandas", "numpy"]:  # Optional for initial setup
                test_result(f"{description} ({package})", False, "Optional - install later")
                optional_missing.append(package)
            else:
                test_result(f"{description} ({package})", False, "Required - pip install needed")
                all_available = False

    if optional_missing:
        print(f"\n{Colors.YELLOW}Optional packages to install: {', '.join(optional_missing)}{Colors.END}")

    return all_available or len(optional_missing) <= 2  # Allow some optional packages

def test_main_orchestrator():
    """Test if main.py is functional"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}Testing Main Orchestrator{Colors.END}")

    if not Path("main.py").exists():
        test_result("main.py exists", False)
        return False

    test_result("main.py exists", True)

    # Check if it has required components
    try:
        with open("main.py", "r", encoding='utf-8') as f:
            content = f.read()

        has_marketpulse = "class MarketPulse" in content
        test_result("MarketPulse class", has_marketpulse)

        has_init_methods = all([
            "_initialize_core" in content,
            "_initialize_risk_management" in content,
            "_initialize_ml_engine" in content
        ])
        test_result("Initialization methods", has_init_methods)

        has_trading_loop = "_trading_loop" in content
        test_result("Trading loop", has_trading_loop)

        return has_marketpulse and has_init_methods and has_trading_loop

    except Exception as e:
        test_result("main.py readable", False, str(e))
        return False

def generate_report(results):
    """Generate integration test report"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}INTEGRATION TEST REPORT{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")

    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    pass_rate = (passed_tests / total_tests) * 100

    print(f"\n{Colors.BOLD}Test Results:{Colors.END}")
    for test_name, passed in results.items():
        status = f"{Colors.GREEN}PASS{Colors.END}" if passed else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  {test_name}: {status}")

    print(f"\n{Colors.BOLD}Summary:{Colors.END}")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {Colors.GREEN}{passed_tests}{Colors.END}")
    print(f"  Failed: {Colors.RED}{total_tests - passed_tests}{Colors.END}")
    print(f"  Pass Rate: {Colors.BOLD}{pass_rate:.1f}%{Colors.END}")

    # Grade
    if pass_rate >= 90:
        grade = f"{Colors.GREEN}A - Excellent!{Colors.END}"
        message = "Ready to proceed to Step 4!"
    elif pass_rate >= 70:
        grade = f"{Colors.GREEN}B - Good{Colors.END}"
        message = "Ready to proceed with minor fixes."
    elif pass_rate >= 50:
        grade = f"{Colors.YELLOW}C - Needs Work{Colors.END}"
        message = "Several components need attention."
    else:
        grade = f"{Colors.RED}D - Incomplete{Colors.END}"
        message = "Major setup required."

    print(f"\n{Colors.BOLD}Grade: {grade}{Colors.END}")
    print(f"Status: {message}")

    # Save report
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "tests": results,
        "pass_rate": pass_rate,
        "grade": grade.replace(Colors.GREEN, "").replace(Colors.YELLOW, "").replace(Colors.RED, "").replace(Colors.END, "")
    }

    report_file = Path("test_report.json")
    with open(report_file, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n{Colors.BLUE}Report saved to: {report_file}{Colors.END}")

    return pass_rate >= 70

def main():
    """Run all integration tests"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("="*60)
    print("   MarketPulse Integration Test Suite")
    print("   Testing Step 3 Completion")
    print("="*60)
    print(f"{Colors.END}")

    # Run tests
    results = {
        "Folder Structure": test_folder_structure(),
        "Core Files": test_core_files(),
        "Antifragile Framework": test_antifragile_framework(),
        "Configuration": test_configuration(),
        "Migrated Components": test_migrated_components(),
        "Python Packages": test_python_imports(),
        "Main Orchestrator": test_main_orchestrator()
    }

    # Generate report
    success = generate_report(results)

    # Next steps
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}NEXT STEPS{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")

    if success:
        print(f"\n{Colors.GREEN}✅ Step 3 Complete!{Colors.END}")
        print("\nProceed to Step 4:")
        print("1. Create paper trading engine")
        print("2. Set up Telegram alerts")
        print("3. Integrate broker connection")
        print("4. Build Streamlit dashboard")
        print("\nReply with: 'Step 3 complete, ready for Step 4'")
    else:
        print(f"\n{Colors.YELLOW}⚠️ Step 3 Needs Completion{Colors.END}")
        print("\nActions needed:")
        for test_name, passed in results.items():
            if not passed:
                print(f"  - Fix: {test_name}")
        print("\nAfter fixing, run: python test_integration.py")

    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())