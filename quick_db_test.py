"""
Quick Database Test - SQLite Compatible
Run this to verify the database setup works correctly
"""

import sys
from pathlib import Path

# Setup paths
current_dir = Path.cwd()
data_dir = current_dir / "06_DATA"
sys.path.insert(0, str(data_dir))


def test_imports():
    """Test if all imports work"""
    try:
        print("🔍 Testing imports...")

        from database.db_setup import Base
        from database.models import MarketData, Trade, Signal, Portfolio
        from sqlalchemy import create_engine

        print("✅ All imports successful!")
        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_database_creation():
    """Test database creation"""
    try:
        print("🗄️ Testing database creation...")

        from sqlalchemy import create_engine
        from database.db_setup import Base

        # Create SQLite database
        db_path = current_dir / "test_marketpulse.db"
        engine = create_engine(f"sqlite:///{db_path}")

        # Create all tables
        Base.metadata.create_all(bind=engine)

        print("✅ Database creation successful!")
        print(f"📁 Database created at: {db_path}")

        return True

    except Exception as e:
        print(f"❌ Database creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_operations():
    """Test basic data operations"""
    try:
        print("📊 Testing data operations...")

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from database.models import MarketData, Portfolio
        from datetime import datetime

        # Connect to test database
        db_path = current_dir / "test_marketpulse.db"
        engine = create_engine(f"sqlite:///{db_path}")
        Session = sessionmaker(bind=engine)
        session = Session()

        # Test 1: Insert market data
        market_data = MarketData(
            symbol='TEST',
            timestamp=datetime.now(),
            timeframe='1d',
            open_price=100.0,
            high_price=105.0,
            low_price=95.0,
            close_price=102.0,
            volume=10000,
            data_source='test'
        )

        session.add(market_data)
        session.commit()
        print("✅ Market data insertion successful")

        # Test 2: Query data
        result = session.query(MarketData).filter(MarketData.symbol == 'TEST').first()
        if result:
            print(f"✅ Data query successful: {result.symbol} @ {result.close_price}")

        # Test 3: Portfolio creation
        portfolio = Portfolio(
            name='Test Portfolio',
            portfolio_type='PAPER',
            total_capital=100000.0,
            available_capital=100000.0,
            invested_capital=0.0
        )

        session.add(portfolio)
        session.commit()
        print("✅ Portfolio creation successful")

        session.close()

        return True

    except Exception as e:
        print(f"❌ Data operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""

    print("🧪 MarketPulse Database Quick Test")
    print("=" * 40)

    if not test_imports():
        print("❌ Fix imports first")
        return False

    if not test_database_creation():
        print("❌ Fix database creation")
        return False

    if not test_data_operations():
        print("❌ Fix data operations")
        return False

    print("\n🎉 All tests passed!")
    print("✅ Database setup is working correctly")
    print("🚀 Ready for the full hybrid setup!")

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n📋 Next step:")
        print("   python hybrid_database_setup.py")
    sys.exit(0 if success else 1)