# 06_DATA/fix_database_schema.py
"""
Quick fix for database schema issues in 2-year ML training system
Resolves index creation conflicts and ensures proper table structure

Location: #06_DATA/fix_database_schema.py
"""

import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_database_schema():
    """Fix database schema by ensuring proper table creation before indexes"""

    db_path = "06_DATA/marketpulse_training.db"

    logger.info(f"Fixing database schema at: {db_path}")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Drop problematic indexes if they exist
        indexes_to_drop = [
            "idx_market_data_symbol_time",
            "idx_technical_symbol_time",
            "idx_ml_predictions_symbol_time",
            "idx_backtest_strategy"
        ]

        for index_name in indexes_to_drop:
            try:
                cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
                logger.info(f"Dropped index: {index_name}")
            except:
                pass

        # Ensure tables exist with proper schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data_enhanced (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                symbol_clean TEXT NOT NULL,
                market_cap_category TEXT NOT NULL,
                sector TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume INTEGER NOT NULL,
                adj_close REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                sma_5 REAL, sma_10 REAL, sma_20 REAL, sma_50 REAL,
                ema_5 REAL, ema_10 REAL, ema_20 REAL, ema_50 REAL,
                rsi_14 REAL, rsi_21 REAL,
                macd REAL, macd_signal REAL, macd_histogram REAL,
                stoch_k REAL, stoch_d REAL,
                williams_r REAL, cci REAL,
                bb_upper REAL, bb_middle REAL, bb_lower REAL, bb_width REAL,
                atr_14 REAL, atr_21 REAL,
                volume_sma_20 REAL, volume_ratio REAL,
                vwap REAL, mfi REAL,
                support_level REAL, resistance_level REAL,
                trend_strength REAL, volatility_regime REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        """)

        # Now create indexes safely
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data_enhanced(symbol, timestamp)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_technical_symbol_time ON technical_indicators(symbol, timestamp)")

        conn.commit()
        conn.close()

        logger.info("Database schema fixed successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to fix database schema: {e}")
        return False


if __name__ == "__main__":
    print("FIXING DATABASE SCHEMA...")

    if fix_database_schema():
        print("SUCCESS: Database schema fixed!")
        print("Now you can run: python 06_DATA/two_year_ml_training_system.py")
    else:
        print("ERROR: Failed to fix database schema")