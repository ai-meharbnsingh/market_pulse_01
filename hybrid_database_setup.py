"""
MarketPulse Hybrid Database Setup
SQLite for immediate use, PostgreSQL-ready for production

This allows you to start immediately without PostgreSQL installation,
then upgrade to PostgreSQL + TimescaleDB when ready.

Usage: python hybrid_database_setup.py
"""

import sys
import os
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add database path
current_dir = Path.cwd()
data_dir = current_dir / "06_DATA"
sys.path.insert(0, str(data_dir))

from database.models import Base, MarketData, Trade, Signal, Portfolio, PortfolioSnapshot

class HybridDatabaseSetup:
    """
    Hybrid database setup: SQLite for development, PostgreSQL for production
    """

    def __init__(self, use_sqlite=True):
        """Initialize with SQLite or PostgreSQL"""

        self.use_sqlite = use_sqlite
        self.db_path = current_dir / "marketpulse.db"

        if use_sqlite:
            # SQLite setup for immediate use
            self.db_url = f"sqlite:///{self.db_path}"
            logger.info("🗃️ Using SQLite database for development")
        else:
            # PostgreSQL setup for production
            db_params = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', 5432)),
                'database': os.getenv('DB_NAME', 'marketpulse'),
                'user': os.getenv('DB_USER', 'marketpulse_user'),
                'password': os.getenv('DB_PASSWORD', 'secure_password_123')
            }
            self.db_url = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
            logger.info("🐘 Using PostgreSQL database for production")

        self.engine = None
        self.SessionLocal = None

    def setup_database(self):
        """Setup database with tables and indexes"""

        try:
            # Create SQLAlchemy engine
            if self.use_sqlite:
                self.engine = create_engine(
                    self.db_url,
                    echo=False,
                    pool_timeout=20,
                    pool_recycle=-1,
                    connect_args={"check_same_thread": False}  # SQLite specific
                )
            else:
                self.engine = create_engine(
                    self.db_url,
                    echo=False,
                    pool_size=10,
                    max_overflow=20,
                    pool_timeout=30,
                    pool_recycle=3600
                )

            # Test connection
            with self.engine.connect() as conn:
                if self.use_sqlite:
                    result = conn.execute(text("SELECT sqlite_version()"))
                    version = result.fetchone()[0]
                    logger.info(f"✅ SQLite connection established: v{version}")
                else:
                    result = conn.execute(text("SELECT version()"))
                    version = result.fetchone()[0]
                    logger.info(f"✅ PostgreSQL connection established: {version[:50]}...")

            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("✅ Database tables created successfully")

            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

            # Add SQLite-specific optimizations
            if self.use_sqlite:
                self._optimize_sqlite()

            logger.info("✅ Database setup completed successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            return False

    def _optimize_sqlite(self):
        """Apply SQLite-specific optimizations"""

        try:
            with self.engine.connect() as conn:
                # Enable WAL mode for better concurrency
                conn.execute(text("PRAGMA journal_mode=WAL"))

                # Optimize for performance
                conn.execute(text("PRAGMA synchronous=NORMAL"))
                conn.execute(text("PRAGMA cache_size=10000"))
                conn.execute(text("PRAGMA temp_store=MEMORY"))

                # Create indexes for better performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data (symbol, timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades (symbol, entry_timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_signals_symbol_confidence ON signals (symbol, confidence)",
                    "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status)",
                ]

                for index_sql in indexes:
                    conn.execute(text(index_sql))

                conn.commit()
                logger.info("✅ SQLite optimizations applied")

        except Exception as e:
            logger.warning(f"⚠️ SQLite optimization failed: {e}")

    def test_database(self):
        """Test database operations"""

        if not self.SessionLocal:
            logger.error("❌ Database not initialized")
            return False

        try:
            session = self.SessionLocal()

            # Test 1: Insert sample market data
            sample_data = MarketData(
                symbol='TEST',
                timestamp=datetime.now(),
                timeframe='1d',
                open_price=1000.0,
                high_price=1020.0,
                low_price=990.0,
                close_price=1010.0,
                volume=100000,
                data_source='test_setup',
                quality_score=1.0
            )

            session.add(sample_data)
            session.commit()
            logger.info("✅ Market data insertion test passed")

            # Test 2: Query data back
            queried = session.query(MarketData).filter(MarketData.symbol == 'TEST').first()
            if queried:
                logger.info(f"✅ Data query test passed: {queried.symbol} @ {queried.close_price}")

            # Test 3: Insert portfolio
            portfolio = Portfolio(
                name='Test Portfolio',
                portfolio_type='PAPER',
                total_capital=100000.0,
                available_capital=100000.0,
                invested_capital=0.0
            )

            session.add(portfolio)
            session.commit()
            logger.info("✅ Portfolio creation test passed")

            # Test 4: Complex query
            recent_data = session.query(MarketData).filter(
                MarketData.timestamp >= datetime.now() - timedelta(hours=1)
            ).count()

            logger.info(f"✅ Complex query test passed: {recent_data} recent records")

            # Cleanup test data
            session.query(MarketData).filter(MarketData.data_source == 'test_setup').delete()
            session.query(Portfolio).filter(Portfolio.name == 'Test Portfolio').delete()
            session.commit()

            session.close()
            logger.info("✅ All database tests passed!")
            return True

        except Exception as e:
            logger.error(f"❌ Database test failed: {e}")
            return False

    def get_session(self):
        """Get database session"""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")
        return self.SessionLocal()

    def get_engine(self):
        """Get database engine"""
        if not self.engine:
            raise RuntimeError("Database not initialized")
        return self.engine

    def get_database_info(self):
        """Get database information"""

        info = {
            'type': 'SQLite' if self.use_sqlite else 'PostgreSQL',
            'url': self.db_url if self.use_sqlite else 'postgresql://[hidden]',
            'file': str(self.db_path) if self.use_sqlite else None,
            'ready': self.engine is not None
        }

        return info

def main():
    """Main setup function"""

    print("🚀 MarketPulse Hybrid Database Setup")
    print("=" * 50)

    # Check if PostgreSQL is available
    postgresql_available = False
    try:
        import psycopg2
        # Try to connect to PostgreSQL
        conn_string = f"host=localhost port=5432 user=postgres"
        conn = psycopg2.connect(conn_string)
        conn.close()
        postgresql_available = True
        print("🐘 PostgreSQL detected and available")
    except:
        print("🗃️ PostgreSQL not available - using SQLite")

    # Setup database
    db_setup = HybridDatabaseSetup(use_sqlite=not postgresql_available)

    print(f"\n📊 Database Type: {db_setup.get_database_info()['type']}")

    # Run setup
    if db_setup.setup_database():
        print("\n🧪 Running database tests...")

        if db_setup.test_database():
            info = db_setup.get_database_info()

            print("\n🎉 Database setup completed successfully!")
            print(f"📊 Type: {info['type']}")

            if info['file']:
                print(f"📁 Location: {info['file']}")

            print("\n✅ MarketPulse database is ready!")
            print("\n📋 What this enables:")
            print("   ✅ Store real market data")
            print("   ✅ Track all trades and performance")
            print("   ✅ ML model predictions and outcomes")
            print("   ✅ Risk management calculations")
            print("   ✅ Complete audit trail")

            print("\n🚀 Next Steps:")
            print("   1. ✅ Database foundation ready")
            print("   2. 📊 Ready for Phase 1, Step 2: Data Pipeline")
            print("   3. 🔄 Connect real market data with yfinance")

            if db_setup.use_sqlite:
                print("\n💡 Future Upgrade:")
                print("   - Install PostgreSQL + TimescaleDB when ready")
                print("   - Run migration script to transfer data")
                print("   - Get enterprise-grade time-series performance")

            return True
        else:
            print("\n❌ Database tests failed!")
            return False
    else:
        print("\n❌ Database setup failed!")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🏆 Ready to proceed to Phase 1, Step 2!")
    else:
        print("\n📞 Need help? Check the troubleshooting guide.")

    sys.exit(0 if success else 1)