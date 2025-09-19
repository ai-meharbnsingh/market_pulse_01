"""
Simple Database Setup Script for MarketPulse
Run this from your MarketPulse project root directory

Usage: python setup_database.py
"""

import sys
import os
from pathlib import Path

# Ensure we're in the right directory
current_dir = Path.cwd()
print(f"Running from: {current_dir}")

# Add the 06_DATA directory to Python path
data_dir = current_dir / "06_DATA"
if not data_dir.exists():
    print("❌ Error: 06_DATA directory not found!")
    print(f"   Expected: {data_dir}")
    print("   Are you running from the MarketPulse project root?")
    sys.exit(1)

sys.path.insert(0, str(data_dir))

try:
    # Test imports first
    print("🔍 Testing imports...")

    print("   Importing psycopg2...")
    import psycopg2

    print("   Importing SQLAlchemy...")
    from sqlalchemy import create_engine

    print("   Importing database components...")
    from database.db_setup import DatabaseSetup
    from database.models import Base, MarketData, Trade, Signal, Portfolio

    print("✅ All imports successful!")

    # Initialize and test database
    print("\n🗄️ Initializing database setup...")
    db_setup = DatabaseSetup()

    print("🔧 Setting up database...")
    success = db_setup.setup_complete_database()

    if success:
        print("✅ Database setup completed!")

        # Test basic operations
        print("\n🧪 Testing basic operations...")
        if db_setup.test_connection():
            print("✅ Database connection test passed!")
            print("\n🎉 Database is ready for MarketPulse!")
            print("\n📋 Next step: Run the trading system!")
        else:
            print("❌ Connection test failed")
    else:
        print("❌ Database setup failed")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n📋 Please install missing dependencies:")
    print("   pip install psycopg2-binary sqlalchemy pyyaml python-dotenv")

except Exception as e:
    print(f"❌ Setup error: {e}")
    print("\n📋 Common issues:")
    print("   1. PostgreSQL not installed/running")
    print("   2. Missing .env file with database credentials")
    print("   3. Incorrect database permissions")