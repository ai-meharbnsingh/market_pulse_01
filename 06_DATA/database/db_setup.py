"""
MarketPulse Database Setup
PostgreSQL + TimescaleDB for Financial Time-Series Data

Expert Recommendation: "PostgreSQL with TimescaleDB extension is the industry standard
for financial data. It will give you highly optimized time-series queries for free."

Location: #06_DATA/database/db_setup.py
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLAlchemy Base for ORM models
Base = declarative_base()


class DatabaseSetup:
    """
    PostgreSQL + TimescaleDB Database Setup Manager

    Features:
    - Automatic PostgreSQL database creation
    - TimescaleDB extension installation
    - Connection pool management
    - Schema validation and migration
    - Performance optimization for time-series queries
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize database setup manager"""

        # Load configuration
        self.config = self._load_config(config_path)

        # Database connection parameters
        self.db_params = {
            'host': os.getenv('DB_HOST', self.config.get('host', 'localhost')),
            'port': int(os.getenv('DB_PORT', self.config.get('port', 5432))),
            'database': os.getenv('DB_NAME', self.config.get('database', 'marketpulse')),
            'user': os.getenv('DB_USER', self.config.get('user', 'marketpulse_user')),
            'password': os.getenv('DB_PASSWORD', self.config.get('password', 'secure_password_123'))
        }

        # Admin connection for database creation
        self.admin_params = {
            'host': self.db_params['host'],
            'port': self.db_params['port'],
            'database': 'postgres',  # Default database for admin operations
            'user': os.getenv('DB_ADMIN_USER', 'postgres'),
            'password': os.getenv('DB_ADMIN_PASSWORD', 'admin_password')
        }

        # SQLAlchemy engine and session
        self.engine = None
        self.SessionLocal = None

        logger.info("Database setup manager initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load database configuration from YAML file"""

        if config_path is None:
            config_path = Path(__file__).parent / "config" / "database_config.yaml"

        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)

        # Default configuration
        return {
            'host': 'localhost',
            'port': 5432,
            'database': 'marketpulse',
            'user': 'marketpulse_user',
            'password': 'secure_password_123',
            'pool_size': 20,
            'max_overflow': 0,
            'pool_timeout': 30,
            'pool_recycle': 3600
        }

    def create_database_and_user(self) -> bool:
        """
        Create PostgreSQL database and user if they don't exist

        Returns:
            bool: True if successful, False otherwise
        """

        try:
            # Connect to PostgreSQL server as admin
            admin_conn = psycopg2.connect(**self.admin_params)
            admin_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = admin_conn.cursor()

            # Create user if not exists
            cursor.execute(f"""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = '{self.db_params['user']}') THEN
                        CREATE USER {self.db_params['user']} WITH PASSWORD '{self.db_params['password']}';
                    END IF;
                END
                $$;
            """)
            logger.info(f"✅ User '{self.db_params['user']}' created/verified")

            # Create database if not exists
            cursor.execute(f"""
                SELECT 1 FROM pg_database WHERE datname = '{self.db_params['database']}'
            """)

            if not cursor.fetchone():
                cursor.execute(f"CREATE DATABASE {self.db_params['database']} OWNER {self.db_params['user']}")
                logger.info(f"✅ Database '{self.db_params['database']}' created")
            else:
                logger.info(f"✅ Database '{self.db_params['database']}' already exists")

            # Grant privileges
            cursor.execute(f"GRANT ALL PRIVILEGES ON DATABASE {self.db_params['database']} TO {self.db_params['user']}")

            cursor.close()
            admin_conn.close()

            return True

        except Exception as e:
            logger.error(f"❌ Failed to create database/user: {e}")
            return False

    def install_timescaledb_extension(self) -> bool:
        """
        Install TimescaleDB extension for time-series optimization

        Returns:
            bool: True if successful, False otherwise
        """

        try:
            # Connect to the target database as admin
            admin_params = self.admin_params.copy()
            admin_params['database'] = self.db_params['database']

            admin_conn = psycopg2.connect(**admin_params)
            admin_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = admin_conn.cursor()

            # Install TimescaleDB extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
            logger.info("✅ TimescaleDB extension installed/verified")

            # Grant necessary permissions to our user
            cursor.execute(f"GRANT ALL ON SCHEMA public TO {self.db_params['user']}")
            cursor.execute(f"GRANT ALL ON ALL TABLES IN SCHEMA public TO {self.db_params['user']}")
            cursor.execute(f"GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO {self.db_params['user']}")
            cursor.execute(f"GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO {self.db_params['user']}")

            cursor.close()
            admin_conn.close()

            return True

        except Exception as e:
            logger.error(f"❌ Failed to install TimescaleDB extension: {e}")
            logger.warning("⚠️ TimescaleDB not available - using standard PostgreSQL")
            return False

    def create_sqlalchemy_engine(self) -> bool:
        """
        Create SQLAlchemy engine with connection pooling

        Returns:
            bool: True if successful, False otherwise
        """

        try:
            # Database connection URL
            db_url = f"postgresql://{self.db_params['user']}:{self.db_params['password']}@{self.db_params['host']}:{self.db_params['port']}/{self.db_params['database']}"

            # Create engine with connection pooling
            self.engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=self.config.get('pool_size', 20),
                max_overflow=self.config.get('max_overflow', 0),
                pool_timeout=self.config.get('pool_timeout', 30),
                pool_recycle=self.config.get('pool_recycle', 3600),
                echo=False  # Set to True for SQL query logging
            )

            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                logger.info(f"✅ PostgreSQL connection established: {version[:50]}...")

            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

            logger.info("✅ SQLAlchemy engine created with connection pooling")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to create SQLAlchemy engine: {e}")
            return False

    def create_schemas_and_tables(self) -> bool:
        """
        Create database schemas and tables from ORM models

        Returns:
            bool: True if successful, False otherwise
        """

        try:
            # Import all ORM models to register them with Base
            from .models import (
                MarketData, Trade, Signal, Portfolio,
                PortfolioSnapshot, AlphaModelPrediction,
                TechnicalIndicator, RiskMetric
            )

            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("✅ Database schemas and tables created")

            # Convert time-series tables to hypertables (TimescaleDB)
            if self._is_timescaledb_available():
                self._create_hypertables()

            return True

        except Exception as e:
            logger.error(f"❌ Failed to create schemas and tables: {e}")
            return False

    def _is_timescaledb_available(self) -> bool:
        """Check if TimescaleDB extension is available"""

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT * FROM pg_extension WHERE extname = 'timescaledb'"))
                return result.fetchone() is not None
        except:
            return False

    def _create_hypertables(self) -> None:
        """Convert time-series tables to TimescaleDB hypertables for optimization"""

        hypertables = [
            ('market_data', 'timestamp'),
            ('trades', 'timestamp'),
            ('signals', 'timestamp'),
            ('portfolio_snapshots', 'timestamp'),
            ('alpha_predictions', 'timestamp'),
            ('technical_indicators', 'timestamp'),
            ('risk_metrics', 'timestamp')
        ]

        try:
            with self.engine.connect() as conn:
                for table_name, time_column in hypertables:
                    try:
                        # Create hypertable (will skip if already exists)
                        conn.execute(text(f"""
                            SELECT create_hypertable('{table_name}', '{time_column}', 
                                                    if_not_exists => TRUE,
                                                    chunk_time_interval => INTERVAL '1 day')
                        """))
                        logger.info(f"✅ Hypertable created: {table_name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not create hypertable {table_name}: {e}")

                conn.commit()

        except Exception as e:
            logger.error(f"❌ Failed to create hypertables: {e}")

    def optimize_database(self) -> None:
        """Apply database optimizations for trading data"""

        optimizations = [
            # Indexes for common queries
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_symbol_time ON market_data (symbol, timestamp DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_symbol_time ON trades (symbol, timestamp DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_symbol_confidence ON signals (symbol, confidence DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_snapshots_time ON portfolio_snapshots (timestamp DESC)",

            # Partial indexes for active trades
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_active ON trades (status) WHERE status = 'ACTIVE'",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_recent ON signals (timestamp) WHERE timestamp >= NOW() - INTERVAL '7 days'",

            # Statistics for query optimization
            "ANALYZE market_data",
            "ANALYZE trades",
            "ANALYZE signals",
            "ANALYZE portfolio_snapshots"
        ]

        try:
            with self.engine.connect() as conn:
                for sql in optimizations:
                    try:
                        conn.execute(text(sql))
                        logger.info(f"✅ Applied optimization: {sql[:50]}...")
                    except Exception as e:
                        logger.warning(f"⚠️ Optimization failed: {sql[:30]}... - {e}")

                conn.commit()
                logger.info("✅ Database optimizations applied")

        except Exception as e:
            logger.error(f"❌ Failed to apply optimizations: {e}")

    def setup_complete_database(self) -> bool:
        """
        Complete database setup process

        Returns:
            bool: True if successful, False otherwise
        """

        logger.info("🚀 Starting complete database setup...")

        # Step 1: Create database and user
        if not self.create_database_and_user():
            logger.error("❌ Database setup failed at user/database creation")
            return False

        # Step 2: Install TimescaleDB extension
        self.install_timescaledb_extension()  # Non-critical, continues without it

        # Step 3: Create SQLAlchemy engine
        if not self.create_sqlalchemy_engine():
            logger.error("❌ Database setup failed at engine creation")
            return False

        # Step 4: Create schemas and tables
        if not self.create_schemas_and_tables():
            logger.error("❌ Database setup failed at schema creation")
            return False

        # Step 5: Apply optimizations
        self.optimize_database()

        logger.info("🎉 Database setup completed successfully!")
        logger.info(f"📊 Database: {self.db_params['database']} on {self.db_params['host']}:{self.db_params['port']}")
        logger.info(f"👤 User: {self.db_params['user']}")
        logger.info(f"⚡ TimescaleDB: {'Enabled' if self._is_timescaledb_available() else 'Disabled'}")

        return True

    def get_session(self):
        """Get database session for ORM operations"""

        if self.SessionLocal is None:
            raise RuntimeError("Database not initialized. Call setup_complete_database() first.")

        return self.SessionLocal()

    def get_engine(self):
        """Get SQLAlchemy engine for direct SQL operations"""

        if self.engine is None:
            raise RuntimeError("Database not initialized. Call setup_complete_database() first.")

        return self.engine

    def test_connection(self) -> bool:
        """Test database connection and basic operations"""

        try:
            with self.engine.connect() as conn:
                # Test basic query
                result = conn.execute(text("SELECT NOW() as current_time"))
                current_time = result.fetchone()[0]
                logger.info(f"✅ Connection test passed: {current_time}")

                # Test TimescaleDB if available
                if self._is_timescaledb_available():
                    result = conn.execute(text("SELECT timescaledb_version()"))
                    ts_version = result.fetchone()[0]
                    logger.info(f"✅ TimescaleDB version: {ts_version}")

                return True

        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False


def main():
    """Main function for standalone database setup"""

    print("🗄️ MarketPulse Database Setup")
    print("=" * 50)

    # Initialize database setup
    db_setup = DatabaseSetup()

    # Setup complete database
    success = db_setup.setup_complete_database()

    if success:
        # Test the connection
        db_setup.test_connection()
        print("\n✅ Database setup completed successfully!")
        print("🚀 Ready for MarketPulse data operations!")
    else:
        print("\n❌ Database setup failed!")
        print("📋 Please check the logs and configuration.")

    return success


if __name__ == "__main__":
    main()