"""
Minimal Database Test - Isolate JSONB Issue
Tests with minimal models to find the exact problem

Usage: python minimal_db_test.py
"""

import sys
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import minimal models
sys.path.append(str(Path.cwd()))
from minimal_models import Base, MarketData, Portfolio, Trade


def test_minimal_database():
    """Test with minimal models only"""

    print("🧪 Minimal Database Test")
    print("=" * 30)

    try:
        # Create SQLite engine
        db_path = Path.cwd() / "minimal_test.db"
        engine = create_engine(f"sqlite:///{db_path}", echo=True)

        print("✅ Engine created")

        # Create tables
        print("📊 Creating tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ Tables created successfully!")

        # Test session
        Session = sessionmaker(bind=engine)
        session = Session()

        # Insert test data
        market_data = MarketData(
            symbol='TEST',
            timestamp=datetime.now(),
            timeframe='1d',
            open_price=100.0,
            high_price=105.0,
            low_price=95.0,
            close_price=102.0,
            volume=10000
        )

        session.add(market_data)
        session.commit()
        print("✅ Data insertion successful")

        # Query data
        result = session.query(MarketData).first()
        if result:
            print(f"✅ Query successful: {result.symbol} @ {result.close_price}")

        session.close()

        print("\n🎉 Minimal test passed!")
        print(f"📁 Database: {db_path}")

        return True

    except Exception as e:
        print(f"❌ Minimal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_minimal_database()

    if success:
        print("\n✅ SQLite works fine with minimal models")
        print("🔍 The issue is in the complex models file")
        print("\n📋 Next steps:")
        print("   1. Check for hidden JSONB imports")
        print("   2. Rebuild models.py from scratch")
        print("   3. Clear Python cache")
    else:
        print("\n❌ Even minimal models fail - SQLAlchemy issue")

    sys.exit(0 if success else 1)