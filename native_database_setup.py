"""
Native SQLite Database Setup - No SQLAlchemy
Works with any Python version, including 3.13

Usage: python native_database_setup.py
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


class NativeDatabaseSetup:
    """Native SQLite database setup without SQLAlchemy"""

    def __init__(self, db_name: str = "marketpulse.db"):
        """Initialize native database"""

        self.db_path = Path.cwd() / db_name
        self.conn = None

        print(f"📊 Using Native SQLite Database: {self.db_path}")

    def connect(self) -> bool:
        """Connect to database"""
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row  # Enable dict-like access
            print("✅ Database connection established")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    def create_tables(self) -> bool:
        """Create all database tables"""

        tables = {
            "market_data": """
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    timeframe TEXT NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    data_source TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """,

            "portfolios": """
                CREATE TABLE IF NOT EXISTS portfolios (
                    portfolio_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    portfolio_type TEXT NOT NULL,
                    total_capital REAL NOT NULL,
                    available_capital REAL NOT NULL,
                    invested_capital REAL NOT NULL,
                    unrealized_pnl REAL DEFAULT 0.0,
                    realized_pnl REAL DEFAULT 0.0,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """,

            "trades": """
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    action TEXT NOT NULL CHECK (action IN ('BUY', 'SELL')),
                    quantity INTEGER NOT NULL CHECK (quantity > 0),
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'CLOSED', 'CANCELLED')),
                    gross_pnl REAL DEFAULT 0.0,
                    commission REAL DEFAULT 0.0,
                    net_pnl REAL DEFAULT 0.0,
                    return_pct REAL DEFAULT 0.0,
                    entry_timestamp DATETIME NOT NULL,
                    exit_timestamp DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """,

            "signals": """
                CREATE TABLE IF NOT EXISTS signals (
                    signal_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL', 'HOLD')),
                    strength REAL NOT NULL CHECK (strength >= 0.0 AND strength <= 1.0),
                    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
                    current_price REAL NOT NULL,
                    predicted_target REAL,
                    predicted_stop_loss REAL,
                    executed INTEGER DEFAULT 0,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """,

            "alpha_predictions": """
                CREATE TABLE IF NOT EXISTS alpha_predictions (
                    prediction_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    features TEXT,  -- JSON as text
                    feature_importance TEXT,  -- JSON as text
                    predicted_direction TEXT,
                    probability_profitable REAL,
                    predicted_return_pct REAL,
                    model_confidence REAL,
                    actual_direction TEXT,
                    actual_return_pct REAL,
                    prediction_accuracy REAL,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """,

            "technical_indicators": """
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    indicator_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    rsi_14 REAL,
                    macd REAL,
                    macd_signal REAL,
                    sma_20 REAL,
                    sma_50 REAL,
                    bb_upper REAL,
                    bb_lower REAL,
                    atr_14 REAL,
                    volume_ratio REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """,

            "system_logs": """
                CREATE TABLE IF NOT EXISTS system_logs (
                    log_id TEXT PRIMARY KEY,
                    level TEXT NOT NULL,
                    module TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    context TEXT,  -- JSON as text
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
        }

        try:
            cursor = self.conn.cursor()

            for table_name, sql in tables.items():
                cursor.execute(sql)
                print(f"✅ Table created: {table_name}")

            self.conn.commit()
            print("✅ All tables created successfully")
            return True

        except Exception as e:
            print(f"❌ Table creation failed: {e}")
            return False

    def create_indexes(self) -> bool:
        """Create database indexes for performance"""

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data (symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades (symbol, entry_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_signals_symbol_confidence ON signals (symbol, confidence)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_symbol_time ON alpha_predictions (symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_technical_symbol_time ON technical_indicators (symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_logs_level_time ON system_logs (level, timestamp)"
        ]

        try:
            cursor = self.conn.cursor()

            for index_sql in indexes:
                cursor.execute(index_sql)

            self.conn.commit()
            print("✅ Database indexes created")
            return True

        except Exception as e:
            print(f"❌ Index creation failed: {e}")
            return False

    def test_database(self) -> bool:
        """Test database operations"""

        try:
            cursor = self.conn.cursor()

            # Test 1: Insert market data
            cursor.execute("""
                INSERT INTO market_data (symbol, timestamp, timeframe, open_price, high_price, 
                                       low_price, close_price, volume, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ('TEST', datetime.now(), '1d', 100.0, 105.0, 95.0, 102.0, 10000, 'native_test'))

            print("✅ Market data insertion successful")

            # Test 2: Query data back
            cursor.execute("SELECT * FROM market_data WHERE symbol = ?", ('TEST',))
            result = cursor.fetchone()
            if result:
                print(f"✅ Data query successful: {result['symbol']} @ {result['close_price']}")

            # Test 3: Insert portfolio
            import uuid
            portfolio_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO portfolios (portfolio_id, name, portfolio_type, total_capital, 
                                      available_capital, invested_capital)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (portfolio_id, 'Test Portfolio', 'PAPER', 100000.0, 100000.0, 0.0))

            print("✅ Portfolio creation successful")

            # Test 4: Complex query
            cursor.execute("""
                SELECT COUNT(*) as recent_count 
                FROM market_data 
                WHERE timestamp >= datetime('now', '-1 hour')
            """)
            count = cursor.fetchone()['recent_count']
            print(f"✅ Complex query successful: {count} recent records")

            self.conn.commit()
            return True

        except Exception as e:
            print(f"❌ Database test failed: {e}")
            return False

    def get_connection(self):
        """Get database connection for direct use"""
        return self.conn

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")


class NativeDatabaseHelper:
    """Helper class for common database operations"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def insert_market_data(self, symbol: str, timestamp: datetime, timeframe: str,
                           open_price: float, high_price: float, low_price: float,
                           close_price: float, volume: int, data_source: str = 'yfinance'):
        """Insert market data"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO market_data 
            (symbol, timestamp, timeframe, open_price, high_price, low_price, 
             close_price, volume, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, timestamp, timeframe, open_price, high_price, low_price,
              close_price, volume, data_source))

        conn.commit()
        conn.close()

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT close_price FROM market_data 
            WHERE symbol = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
        """, (symbol,))

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None

    def insert_trade(self, trade_data: Dict[str, Any]):
        """Insert trade record"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO trades 
            (trade_id, symbol, strategy, action, quantity, entry_price, status, entry_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_data['trade_id'], trade_data['symbol'], trade_data['strategy'],
            trade_data['action'], trade_data['quantity'], trade_data['entry_price'],
            trade_data['status'], trade_data['entry_timestamp']
        ))

        conn.commit()
        conn.close()

    def get_portfolio_value(self, portfolio_id: str) -> Optional[float]:
        """Get current portfolio value"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT total_capital + realized_pnl + unrealized_pnl as total_value
            FROM portfolios 
            WHERE portfolio_id = ?
        """, (portfolio_id,))

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None


def main():
    """Main setup function"""

    print("🚀 MarketPulse Native Database Setup")
    print("=" * 50)
    print("✅ No SQLAlchemy - Works with Python 3.13!")

    # Setup database
    db = NativeDatabaseSetup()

    if not db.connect():
        return False

    if not db.create_tables():
        return False

    if not db.create_indexes():
        return False

    if not db.test_database():
        return False

    print("\n🎉 Native database setup completed successfully!")
    print(f"📁 Database location: {db.db_path}")

    # Show helper usage
    print("\n📋 Database Helper Usage:")
    print("```python")
    print("from native_database_setup import NativeDatabaseHelper")
    print("helper = NativeDatabaseHelper('marketpulse.db')")
    print("helper.insert_market_data('RELIANCE', datetime.now(), '1d', 2500, 2520, 2490, 2510, 1000000)")
    print("price = helper.get_latest_price('RELIANCE')")
    print("```")

    print("\n🎯 What this enables:")
    print("   ✅ Store real market data from yfinance")
    print("   ✅ Track all trades and portfolio performance")
    print("   ✅ ML prediction storage (JSON as text)")
    print("   ✅ Complete trading system database")
    print("   ✅ No compatibility issues with any Python version")

    print("\n🚀 Ready for Phase 1, Step 2: Data Pipeline Creation!")

    db.close()
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed - check the error messages above")
    exit(0 if success else 1)