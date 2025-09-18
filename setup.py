"""
MarketPulse Setup Script
Automates virtual environment creation and dependency installation
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, shell=True):
    """Execute a shell command"""
    try:
        result = subprocess.run(command, shell=shell, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {command}")
            return True
        else:
            print(f"❌ Failed: {command}")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def setup_virtual_environment():
    """Create and activate virtual environment"""
    print("\n" + "=" * 60)
    print("Setting up Virtual Environment")
    print("=" * 60)

    # Determine OS
    is_windows = platform.system() == "Windows"

    # Create virtual environment
    if run_command(f"{sys.executable} -m venv venv"):
        print("✅ Virtual environment created")
    else:
        print("❌ Failed to create virtual environment")
        return False

    # Activation instruction
    if is_windows:
        activate_cmd = "venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"

    print(f"\n📌 To activate the virtual environment, run:")
    print(f"   {activate_cmd}")

    return True


def install_dependencies():
    """Install required packages"""
    print("\n" + "=" * 60)
    print("Installing Dependencies")
    print("=" * 60)

    # Upgrade pip first
    print("\n1. Upgrading pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")

    # Core dependencies first
    print("\n2. Installing core dependencies...")
    core_deps = [
        "python-dotenv",
        "pyyaml",
        "pandas",
        "numpy",
        "requests",
        "aiohttp"
    ]

    for dep in core_deps:
        if not run_command(f"pip install {dep}"):
            print(f"⚠️ Warning: Failed to install {dep}")

    # AI SDKs
    print("\n3. Installing AI SDKs...")
    ai_deps = ["openai", "anthropic", "google-generativeai"]
    for dep in ai_deps:
        if not run_command(f"pip install {dep}"):
            print(f"⚠️ Warning: Failed to install {dep}")

    # ML libraries
    print("\n4. Installing ML libraries...")
    ml_deps = ["scikit-learn", "xgboost", "lightgbm", "joblib"]
    for dep in ml_deps:
        if not run_command(f"pip install {dep}"):
            print(f"⚠️ Warning: Failed to install {dep}")

    # Note about TA-Lib
    print("\n⚠️ NOTE: TA-Lib requires separate installation:")
    print("   Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
    print("   Then: pip install TA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl")
    print("   Linux/Mac: brew install ta-lib (or apt-get install ta-lib)")
    print("   Then: pip install ta-lib")

    # Dashboard
    print("\n5. Installing Streamlit...")
    run_command("pip install streamlit plotly")

    return True


def create_folder_structure():
    """Create any missing folders"""
    print("\n" + "=" * 60)
    print("Creating Folder Structure")
    print("=" * 60)

    folders = [
        "01_CORE/config",
        "01_CORE/logging",
        "02_ANALYSIS/technical",
        "02_ANALYSIS/fundamental",
        "02_ANALYSIS/sentiment",
        "03_ML_ENGINE/models",
        "03_ML_ENGINE/features",
        "03_ML_ENGINE/backtesting",
        "03_ML_ENGINE/retraining",
        "04_RISK",
        "05_EXECUTION/paper_trading",
        "05_EXECUTION/broker",
        "05_EXECUTION/alerts",
        "06_DATA/streaming",
        "06_DATA/historical",
        "06_DATA/database",
        "07_DASHBOARD/pages",
        "07_DASHBOARD/components",
        "08_TESTS",
        "09_DOCS",
        "10_DATA_STORAGE/models",
        "10_DATA_STORAGE/logs",
        "10_DATA_STORAGE/backtest_results"
    ]

    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✅ {folder}")

    return True


def create_env_file():
    """Create .env file from template"""
    print("\n" + "=" * 60)
    print("Creating Environment File")
    print("=" * 60)

    env_template = """# MarketPulse Environment Variables
# COPY THIS TO .env AND FILL WITH YOUR ACTUAL KEYS

# AI Providers
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_claude_key_here
GOOGLE_API_KEY=your_gemini_key_here

# Broker (Zerodha Kite)
KITE_API_KEY=your_kite_api_key
KITE_API_SECRET=your_kite_secret
KITE_ACCESS_TOKEN=your_access_token

# Database (PostgreSQL)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=marketpulse
DB_USER=postgres
DB_PASSWORD=your_password

# Telegram Bot (Optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Data Providers (Optional)
ALPHA_VANTAGE_KEY=your_key
YAHOO_FINANCE_KEY=free
"""

    # Create .env.example
    with open(".env.example", "w") as f:
        f.write(env_template)
    print("✅ Created .env.example")

    # Check if .env exists
    if not Path(".env").exists():
        with open(".env", "w") as f:
            f.write(env_template)
        print("✅ Created .env (remember to add your API keys!)")
    else:
        print("ℹ️ .env already exists")

    return True


def verify_installation():
    """Verify key imports work"""
    print("\n" + "=" * 60)
    print("Verifying Installation")
    print("=" * 60)

    # Test imports
    test_imports = [
        ("pandas", "Data processing"),
        ("numpy", "Numerical computing"),
        ("sklearn", "Machine learning"),
        ("streamlit", "Dashboard"),
        ("yaml", "Configuration"),
    ]

    for module, description in test_imports:
        try:
            __import__(module)
            print(f"✅ {description}: {module}")
        except ImportError:
            print(f"❌ {description}: {module} - NOT INSTALLED")

    # Check for Antifragile Framework
    if Path("01_CORE/antifragile/__init__.py").exists():
        print("✅ Antifragile Framework: FOUND")
    else:
        print("⚠️ Antifragile Framework: NOT FOUND (run migration script)")

    return True


def main():
    """Main setup process"""
    print("=" * 60)
    print("   MarketPulse v2.0 Setup Script")
    print("   Personal AI-Powered Trading System")
    print("=" * 60)

    # Check Python version
    print(f"\nPython Version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required!")
        return

    # Run setup steps
    steps = [
        ("Creating folder structure", create_folder_structure),
        ("Setting up virtual environment", setup_virtual_environment),
        ("Creating environment file", create_env_file),
    ]

    for step_name, step_func in steps:
        print(f"\n🔧 {step_name}...")
        if not step_func():
            print(f"❌ Failed at: {step_name}")
            return

    # Installation instructions
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\n1. Activate virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")

    print("\n2. Install dependencies:")
    print("   pip install -r requirements.txt")

    print("\n3. Add your API keys to .env file")

    print("\n4. Run migration script:")
    print("   python migrate_existing_code.py")

    print("\n5. Test the system:")
    print("   python test_integration.py")

    print("\n6. Start paper trading:")
    print("   python main.py --mode paper")

    print("\n" + "=" * 60)
    print("Setup script completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()