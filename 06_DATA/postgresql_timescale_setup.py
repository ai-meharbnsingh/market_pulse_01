# 06_DATA/postgresql_timescale_setup.py
"""
PostgreSQL + TimescaleDB Setup for Production Financial Data
Industry standard time-series database for high-frequency trading data

Features:
- TimescaleDB extension for time-series optimization
- Proper hypertables for market data
- Partitioning strategies for performance
- Continuous aggregates for real-time analytics
- Compression policies for storage efficiency
- Connection pooling and performance tuning
- Migration utilities from SQLite to PostgreSQL

Location: #06_DATA/postgresql_timescale_setup.py
"""

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import threading
from contextlib import contextmanager
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimescaleDBError(Exception):
    """Custom exception for TimescaleDB related errors"""
    pass


class PostgreSQLTimescaleSetup:
    """Production PostgreSQL + TimescaleDB setup for financial data"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize PostgreSQL + TimescaleDB setup

        Args:
            config: Database configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.connection_pool = None
        self.lock = threading.Lock()

        logger.info("PostgreSQL + TimescaleDB setup initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default PostgreSQL configuration"""
        return {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'marketpulse_prod'),
            'user': os.getenv('POSTGRES_USER', 'marketpulse'),
            'password': os.getenv('POSTGRES_PASSWORD', 'secure_password_here'),
            'pool_min': 2,
            'pool_max': 20,
            'application_name': 'MarketPulse_Production'
        }

    def create_database_and_user(self, admin_config: Dict[str, Any] = None):
        """Create database and user with proper permissions"""
        admin_config = admin_config or {
            'host': self.config['host'],
            'port': self.config['port'],
            'database': 'postgres',  # Connect to default database
            'user': 'postgres',
            'password': os.getenv('POSTGRES_ADMIN_PASSWORD', 'admin_password')
        }

        try:
            # Connect as admin user
            conn = psycopg2.connect(**admin_config)
            conn.autocommit = True
            cursor = conn.cursor()

            # Create user if not exists
            cursor.execute(f"""
                SELECT 1 FROM pg_roles WHERE rolname = '{self.config['user']}'
            """)
            if not cursor.fetchone():
                cursor.execute(f"""
                    CREATE USER {self.config['user']} 
                    WITH PASSWORD '{self.config['password']}'
                """)
                logger.info(f"Created user: {self.config['user']}")

            # Create database if not exists
            cursor.execute(f"""
                SELECT 1 FROM pg_database WHERE datname = '{self.config['database']}'
            """)
            if not cursor.fetchone():
                cursor.execute(f"""
                    CREATE DATABASE {self.config['database']} 
                    OWNER {self.config['user']}
                """)
                logger.info(f"Created database: {self.config['database']}")

            # Grant permissions
            cursor.execute(f"""
                GRANT ALL PRIVILEGES ON DATABASE {self.config['database']} 
                TO {self.config['user']}
            """)

            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to create database/user: {e}")
            raise TimescaleDBError(f"Database setup failed: {e}")

    def initialize_connection_pool(self):
        """Initialize PostgreSQL connection pool"""
        try:
            dsn = (f"host={self.config['host']} "
                   f"port={self.config['port']} "
                   f"dbname={self.config['database']} "
                   f"user={self.config['user']} "
                   f"password={self.config['password']} "
                   f"application_name={self.config['application_name']}")

            self.connection_pool = ThreadedConnectionPool(
                minconn=self.config['pool_min'],
                maxconn=self.config['pool_max'],
                dsn=dsn
            )

            logger.info("PostgreSQL connection pool initialized")

        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise TimescaleDBError(f"Connection pool setup failed: {e}")

    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        if not self.connection_pool:
            self.initialize_connection_pool()

        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def install_timescaledb_extension(self):
        """Install TimescaleDB extension"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Check if TimescaleDB is already installed
                cursor.execute("""
                    SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
                """)

                if not cursor.fetchone():
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
                    conn.commit()
                    logger.info("TimescaleDB extension installed")
                else:
                    logger.info("TimescaleDB extension already installed")

                cursor.close()

        except Exception as e:
            logger.error(f"Failed to install TimescaleDB: {e}")
            raise TimescaleDBError(f"TimescaleDB installation failed: {e}")

    def create_production_schema(self):
        """Create production database schema with TimescaleDB hypertables"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Create schemas
                cursor.execute("CREATE SCHEMA IF NOT EXISTS market_data")
                cursor.execute("CREATE SCHEMA IF NOT EXISTS trading")
                cursor.execute("CREATE SCHEMA IF NOT EXISTS analytics")
                cursor.execute("CREATE SCHEMA IF NOT EXISTS monitoring")

                # Market data hypertable (main time-series table)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_data.ohlcv (
                        time TIMESTAMPTZ NOT NULL,
                        symbol TEXT NOT NULL,
                        open_price NUMERIC(15,6) NOT NULL,
                        high_price NUMERIC(15,6) NOT NULL,
                        low_price NUMERIC(15,6) NOT NULL,
                        close_price NUMERIC(15,6) NOT NULL,
                        volume BIGINT NOT NULL,
                        adj_close NUMERIC(15,6),
                        source TEXT NOT NULL DEFAULT 'yahoo_finance',
                        interval_type TEXT NOT NULL DEFAULT '1d',
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE(time, symbol, interval_type, source)
                    )
                """)

                # Convert to hypertable
                cursor.execute("""
                    SELECT create_hypertable(
                        'market_data.ohlcv', 
                        'time',
                        chunk_time_interval => INTERVAL '1 day',
                        if_not_exists => TRUE
                    )
                """)

                # Real-time quotes table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_data.real_time_quotes (
                        symbol TEXT PRIMARY KEY,
                        current_price NUMERIC(15,6) NOT NULL,
                        bid NUMERIC(15,6),
                        ask NUMERIC(15,6),
                        bid_size INTEGER,
                        ask_size INTEGER,
                        day_change NUMERIC(15,6),
                        day_change_percent NUMERIC(8,4),
                        day_high NUMERIC(15,6),
                        day_low NUMERIC(15,6),
                        day_volume BIGINT,
                        market_cap BIGINT,
                        pe_ratio NUMERIC(8,2),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Trading tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trading.orders (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        symbol TEXT NOT NULL,
                        order_type TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        price NUMERIC(15,6),
                        stop_price NUMERIC(15,6),
                        status TEXT NOT NULL DEFAULT 'pending',
                        filled_quantity INTEGER DEFAULT 0,
                        filled_price NUMERIC(15,6),
                        commission NUMERIC(10,4) DEFAULT 0,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trading.positions (
                        symbol TEXT PRIMARY KEY,
                        quantity INTEGER NOT NULL,
                        avg_cost NUMERIC(15,6) NOT NULL,
                        current_value NUMERIC(15,2),
                        unrealized_pnl NUMERIC(15,2),
                        realized_pnl NUMERIC(15,2) DEFAULT 0,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Analytics tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analytics.portfolio_performance (
                        time TIMESTAMPTZ NOT NULL,
                        total_value NUMERIC(15,2) NOT NULL,
                        daily_pnl NUMERIC(15,2),
                        total_pnl NUMERIC(15,2),
                        cash_balance NUMERIC(15,2),
                        positions_value NUMERIC(15,2),
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Convert portfolio performance to hypertable
                cursor.execute("""
                    SELECT create_hypertable(
                        'analytics.portfolio_performance', 
                        'time',
                        chunk_time_interval => INTERVAL '1 week',
                        if_not_exists => TRUE
                    )
                """)

                # Monitoring tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS monitoring.system_health (
                        time TIMESTAMPTZ NOT NULL,
                        component TEXT NOT NULL,
                        status TEXT NOT NULL,
                        response_time_ms NUMERIC(8,2),
                        error_rate NUMERIC(5,4),
                        details JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                cursor.execute("""
                    SELECT create_hypertable(
                        'monitoring.system_health', 
                        'time',
                        chunk_time_interval => INTERVAL '1 hour',
                        if_not_exists => TRUE
                    )
                """)

                conn.commit()
                cursor.close()

                logger.info("Production schema created successfully")

        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            raise TimescaleDBError(f"Schema creation failed: {e}")

    def create_indexes_and_constraints(self):
        """Create performance indexes and constraints"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Market data indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time 
                    ON market_data.ohlcv (symbol, time DESC)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_interval 
                    ON market_data.ohlcv (symbol, interval_type, time DESC)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_quotes_updated 
                    ON market_data.real_time_quotes (updated_at DESC)
                """)

                # Trading indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_orders_symbol_time 
                    ON trading.orders (symbol, created_at DESC)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_orders_status 
                    ON trading.orders (status, created_at DESC)
                """)

                # Analytics indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_portfolio_time 
                    ON analytics.portfolio_performance (time DESC)
                """)

                # Monitoring indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_health_component_time 
                    ON monitoring.system_health (component, time DESC)
                """)

                conn.commit()
                cursor.close()

                logger.info("Indexes and constraints created")

        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            raise TimescaleDBError(f"Index creation failed: {e}")

    def setup_continuous_aggregates(self):
        """Set up continuous aggregates for real-time analytics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # 1-minute OHLCV aggregates
                cursor.execute("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.ohlcv_1min
                    WITH (timescaledb.continuous) AS
                    SELECT 
                        time_bucket('1 minute', time) AS bucket,
                        symbol,
                        first(open_price, time) AS open_price,
                        max(high_price) AS high_price,
                        min(low_price) AS low_price,
                        last(close_price, time) AS close_price,
                        sum(volume) AS volume
                    FROM market_data.ohlcv
                    GROUP BY bucket, symbol
                    WITH NO DATA
                """)

                # 1-hour OHLCV aggregates
                cursor.execute("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.ohlcv_1hour
                    WITH (timescaledb.continuous) AS
                    SELECT 
                        time_bucket('1 hour', time) AS bucket,
                        symbol,
                        first(open_price, time) AS open_price,
                        max(high_price) AS high_price,
                        min(low_price) AS low_price,
                        last(close_price, time) AS close_price,
                        sum(volume) AS volume
                    FROM market_data.ohlcv
                    GROUP BY bucket, symbol
                    WITH NO DATA
                """)

                # Daily portfolio performance summary
                cursor.execute("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.daily_performance
                    WITH (timescaledb.continuous) AS
                    SELECT 
                        time_bucket('1 day', time) AS bucket,
                        first(total_value, time) AS open_value,
                        max(total_value) AS high_value,
                        min(total_value) AS low_value,
                        last(total_value, time) AS close_value,
                        sum(daily_pnl) AS total_daily_pnl
                    FROM analytics.portfolio_performance
                    GROUP BY bucket
                    WITH NO DATA
                """)

                conn.commit()
                cursor.close()

                logger.info("Continuous aggregates created")

        except Exception as e:
            logger.error(f"Failed to create continuous aggregates: {e}")
            raise TimescaleDBError(f"Continuous aggregate setup failed: {e}")

    def setup_retention_policies(self):
        """Set up data retention and compression policies"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Compress data older than 7 days
                cursor.execute("""
                    SELECT add_compression_policy(
                        'market_data.ohlcv', 
                        INTERVAL '7 days',
                        if_not_exists => TRUE
                    )
                """)

                # Compress monitoring data older than 1 day
                cursor.execute("""
                    SELECT add_compression_policy(
                        'monitoring.system_health', 
                        INTERVAL '1 day',
                        if_not_exists => TRUE
                    )
                """)

                # Drop monitoring data older than 30 days
                cursor.execute("""
                    SELECT add_retention_policy(
                        'monitoring.system_health', 
                        INTERVAL '30 days',
                        if_not_exists => TRUE
                    )
                """)

                conn.commit()
                cursor.close()

                logger.info("Retention and compression policies set up")

        except Exception as e:
            logger.error(f"Failed to set up policies: {e}")
            logger.warning("Compression/retention policies require TimescaleDB licensing")

    def migrate_from_sqlite(self, sqlite_db_path: str):
        """Migrate data from SQLite to PostgreSQL"""
        try:
            # Connect to SQLite
            sqlite_conn = sqlite3.connect(sqlite_db_path)
            sqlite_conn.row_factory = sqlite3.Row

            with self.get_connection() as pg_conn:
                pg_cursor = pg_conn.cursor()

                # Migrate market data
                sqlite_cursor = sqlite_conn.execute("""
                    SELECT * FROM market_data_live ORDER BY timestamp
                """)

                market_data_rows = []
                for row in sqlite_cursor:
                    market_data_rows.append((
                        row['timestamp'], row['symbol'], row['open_price'],
                        row['high_price'], row['low_price'], row['close_price'],
                        row['volume'], row.get('adj_close'), row.get('source', 'yahoo_finance')
                    ))

                if market_data_rows:
                    psycopg2.extras.execute_batch(
                        pg_cursor,
                        """
                        INSERT INTO market_data.ohlcv 
                        (time, symbol, open_price, high_price, low_price, 
                         close_price, volume, adj_close, source)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (time, symbol, interval_type, source) DO NOTHING
                        """,
                        market_data_rows
                    )

                # Migrate real-time quotes
                sqlite_cursor = sqlite_conn.execute("""
                    SELECT * FROM real_time_quotes
                """)

                quote_rows = []
                for row in sqlite_cursor:
                    quote_rows.append((
                        row['symbol'], row['current_price'], row.get('bid'),
                        row.get('ask'), row.get('bid_size'), row.get('ask_size'),
                        row.get('day_change'), row.get('day_change_percent'),
                        row.get('day_high'), row.get('day_low'), row.get('day_volume'),
                        row.get('market_cap'), row.get('pe_ratio'), row.get('updated_at')
                    ))

                if quote_rows:
                    psycopg2.extras.execute_batch(
                        pg_cursor,
                        """
                        INSERT INTO market_data.real_time_quotes 
                        (symbol, current_price, bid, ask, bid_size, ask_size,
                         day_change, day_change_percent, day_high, day_low,
                         day_volume, market_cap, pe_ratio, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol) DO UPDATE SET
                        current_price = EXCLUDED.current_price,
                        updated_at = EXCLUDED.updated_at
                        """,
                        quote_rows
                    )

                pg_conn.commit()

            sqlite_conn.close()

            logger.info(f"Migration from {sqlite_db_path} completed successfully")

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise TimescaleDBError(f"Data migration failed: {e}")

    def test_connection_and_performance(self):
        """Test database connection and performance"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Test basic connection
                cursor.execute("SELECT version(), now()")
                version, current_time = cursor.fetchone()

                # Test TimescaleDB
                cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
                timescale_version = cursor.fetchone()

                # Test hypertable
                cursor.execute("""
                    SELECT hypertable_name, num_chunks 
                    FROM timescaledb_information.hypertables 
                    WHERE hypertable_name = 'ohlcv'
                """)
                hypertable_info = cursor.fetchone()

                # Performance test - insert sample data
                start_time = time.time()
                test_data = [(
                    datetime.now() - timedelta(minutes=i),
                    'TEST',
                    100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i,
                    1000 + i, 100.5 + i, 'test'
                ) for i in range(1000)]

                psycopg2.extras.execute_batch(
                    cursor,
                    """
                    INSERT INTO market_data.ohlcv 
                    (time, symbol, open_price, high_price, low_price, 
                     close_price, volume, adj_close, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    test_data
                )

                insert_time = time.time() - start_time

                # Query performance test
                start_time = time.time()
                cursor.execute("""
                    SELECT symbol, count(*), avg(close_price)
                    FROM market_data.ohlcv 
                    WHERE symbol = 'TEST'
                    GROUP BY symbol
                """)
                results = cursor.fetchall()
                query_time = time.time() - start_time

                # Cleanup test data
                cursor.execute("DELETE FROM market_data.ohlcv WHERE symbol = 'TEST'")
                conn.commit()

                cursor.close()

                test_results = {
                    'postgresql_version': version,
                    'timescaledb_version': timescale_version[0] if timescale_version else None,
                    'hypertable_chunks': hypertable_info[1] if hypertable_info else 0,
                    'insert_performance_ms': round(insert_time * 1000, 2),
                    'query_performance_ms': round(query_time * 1000, 2),
                    'connection_pool_status': 'active',
                    'current_time': current_time
                }

                logger.info("Database connection and performance test completed")
                return test_results

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            raise TimescaleDBError(f"Database test failed: {e}")

    def setup_complete_production_database(self, migrate_sqlite_path: str = None):
        """Complete setup process for production database"""
        logger.info("Starting complete PostgreSQL + TimescaleDB setup...")

        try:
            # Step 1: Initialize connection pool
            self.initialize_connection_pool()

            # Step 2: Install TimescaleDB extension
            self.install_timescaledb_extension()

            # Step 3: Create production schema
            self.create_production_schema()

            # Step 4: Create indexes and constraints
            self.create_indexes_and_constraints()

            # Step 5: Set up continuous aggregates
            self.setup_continuous_aggregates()

            # Step 6: Set up retention policies (optional)
            try:
                self.setup_retention_policies()
            except Exception as e:
                logger.warning(f"Retention policies setup failed (may require license): {e}")

            # Step 7: Migrate existing data if provided
            if migrate_sqlite_path and os.path.exists(migrate_sqlite_path):
                self.migrate_from_sqlite(migrate_sqlite_path)

            # Step 8: Test everything
            test_results = self.test_connection_and_performance()

            logger.info("PostgreSQL + TimescaleDB setup completed successfully!")
            return test_results

        except Exception as e:
            logger.error(f"Complete setup failed: {e}")
            raise TimescaleDBError(f"Production database setup failed: {e}")

    def cleanup(self):
        """Clean up connections"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Connection pool closed")


def create_env_template():
    """Create environment template for PostgreSQL configuration"""
    env_template = """
# PostgreSQL Configuration for MarketPulse Production
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=marketpulse_prod
POSTGRES_USER=marketpulse
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_ADMIN_PASSWORD=postgres_admin_password

# Connection Pool Settings
POSTGRES_POOL_MIN=2
POSTGRES_POOL_MAX=20
"""

    with open('.env.postgresql', 'w') as f:
        f.write(env_template)

    print("Created .env.postgresql template - please update with your credentials")


if __name__ == "__main__":
    # Example usage and testing
    print("PostgreSQL + TimescaleDB Setup for MarketPulse")

    # Create environment template
    create_env_template()

    # Initialize setup (would require actual PostgreSQL instance)
    try:
        setup = PostgreSQLTimescaleSetup()
        print("Setup initialized - ready for production database configuration")

        # Note: Actual setup requires PostgreSQL + TimescaleDB installation
        print("\nNext steps:")
        print("1. Install PostgreSQL 14+ with TimescaleDB extension")
        print("2. Update .env.postgresql with your database credentials")
        print("3. Run setup.setup_complete_production_database()")

    except Exception as e:
        print(f"Setup initialization note: {e}")
        print("This requires actual PostgreSQL installation to complete")