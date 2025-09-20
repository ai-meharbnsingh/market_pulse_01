# 06_DATA/indian_stock_data_fetcher_database_fix.py
"""
CRITICAL FIX: Database Path Synchronization for Indian Stock Data Fetcher
Fixes the disconnect between data writing and reading causing 0 records in statistics

Issue: Data fetcher writes to one database, statistics read from another
Solution: Ensure consistent database path usage throughout

Location: #06_DATA/indian_stock_data_fetcher_database_fix.py
"""

import sqlite3
import os
import logging
from datetime import datetime
from typing import Dict, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabasePathValidator:
    """Validates and fixes database path consistency issues"""

    def __init__(self, base_path: str = "/home/claude"):
        self.base_path = base_path

        # Define the target database paths as per your architecture
        self.target_dbs = {
            'production': 'marketpulse_production.db',
            'training': '06_DATA/marketpulse_training.db',
            'performance': '10_DATA_STORAGE/marketpulse_performance.db'
        }

    def diagnose_database_issue(self) -> Dict:
        """Diagnose the current database path issues"""

        logger.info("🔍 DIAGNOSING DATABASE PATH ISSUES...")

        diagnosis = {
            'database_files_found': [],
            'table_analysis': {},
            'record_counts': {},
            'issue_summary': []
        }

        # Find all .db files in the project
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.db') and not file.startswith('test_'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.base_path)
                    diagnosis['database_files_found'].append(rel_path)

        logger.info(f"Found {len(diagnosis['database_files_found'])} database files")

        # Analyze each database for Indian stock data
        for db_path in diagnosis['database_files_found']:
            full_path = os.path.join(self.base_path, db_path)

            try:
                conn = sqlite3.connect(full_path)
                cursor = conn.cursor()

                # Check for Indian stock training tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

                diagnosis['table_analysis'][db_path] = tables

                # Check for Indian stock data specifically
                indian_data_count = 0
                if 'market_data_enhanced' in tables:
                    try:
                        cursor.execute("SELECT COUNT(*) FROM market_data_enhanced WHERE symbol LIKE '%.NS'")
                        indian_data_count = cursor.fetchone()[0]
                    except:
                        indian_data_count = 0
                elif 'market_data' in tables:
                    try:
                        cursor.execute("SELECT COUNT(*) FROM market_data WHERE symbol LIKE '%.NS'")
                        indian_data_count = cursor.fetchone()[0]
                    except:
                        indian_data_count = 0

                diagnosis['record_counts'][db_path] = indian_data_count

                conn.close()

                if indian_data_count > 0:
                    logger.info(f"✅ Found {indian_data_count} Indian stock records in {db_path}")

            except Exception as e:
                logger.warning(f"⚠️ Could not analyze {db_path}: {e}")
                diagnosis['record_counts'][db_path] = f"Error: {e}"

        # Create issue summary
        total_records = sum(count for count in diagnosis['record_counts'].values() if isinstance(count, int))

        if total_records == 0:
            diagnosis['issue_summary'].append("CRITICAL: No Indian stock data found in any database")
        elif len([db for db, count in diagnosis['record_counts'].items() if isinstance(count, int) and count > 0]) > 1:
            diagnosis['issue_summary'].append("WARNING: Indian stock data scattered across multiple databases")

        return diagnosis

    def fix_database_paths(self) -> bool:
        """Fix database path issues in the Indian stock fetcher"""

        logger.info("🔧 FIXING DATABASE PATH ISSUES...")

        # First, ensure target directories exist
        os.makedirs(os.path.join(self.base_path, '06_DATA'), exist_ok=True)
        os.makedirs(os.path.join(self.base_path, '10_DATA_STORAGE'), exist_ok=True)

        # Create the training database with proper schema
        training_db_path = os.path.join(self.base_path, self.target_dbs['training'])

        conn = sqlite3.connect(training_db_path)
        cursor = conn.cursor()

        # Create the enhanced market data table that the fetcher expects
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

        # Create indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
            ON market_data_enhanced(symbol, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_cap_sector 
            ON market_data_enhanced(market_cap_category, sector)
        """)

        conn.commit()
        conn.close()

        logger.info(f"✅ Created/verified training database: {training_db_path}")

        return True

    def create_fixed_data_fetcher(self) -> str:
        """Create a corrected version of the Indian stock data fetcher"""

        fixed_fetcher_code = '''# 06_DATA/indian_stock_data_fetcher_corrected.py
"""
CORRECTED Indian Stock Data Fetcher
Fixes database path and persistence issues

Key Fixes:
1. Consistent database path usage
2. Proper error handling for data writing
3. Verification of data persistence
4. Better transaction handling

Location: #06_DATA/indian_stock_data_fetcher_corrected.py
"""

import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple, Optional
import time
import numpy as np
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FixedIndianStockDataFetcher:
    """Fixed data fetcher with proper database persistence for Indian stocks"""

    def __init__(self, db_path: str = "06_DATA/marketpulse_training.db"):
        # Ensure absolute path for consistency
        if not os.path.isabs(db_path):
            self.db_path = os.path.abspath(db_path)
        else:
            self.db_path = db_path

        self.indian_universe = self._create_indian_universe()

        # Ensure database directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        logger.info(f"📁 Database path: {self.db_path}")

    def _create_indian_universe(self) -> Dict[str, Dict[str, List[str]]]:
        """Create comprehensive Indian stock universe"""
        return {
            'large_cap': {
                'banking_finance': [
                    'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS',
                    'AXISBANK.NS', 'INDUSINDBK.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS'
                ],
                'information_technology': [
                    'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS'
                ],
                'energy_oil_gas': [
                    'RELIANCE.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS', 'HINDPETRO.NS'
                ],
                'automobile': [
                    'MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS'
                ],
                'pharmaceutical': [
                    'SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'BIOCON.NS'
                ]
            },
            'mid_cap': {
                'telecom': [
                    'BHARTIARTL.NS', 'IDEA.NS'
                ],
                'fmcg': [
                    'HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS'
                ],
                'infrastructure': [
                    'LT.NS', 'ULTRACEMCO.NS', 'GRASIM.NS', 'SHREECEM.NS'
                ]
            }
        }

    def create_training_database_schema(self):
        """Create enhanced database schema for Indian stock training data"""

        logger.info(f"🗂️ Creating database schema at: {self.db_path}")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create enhanced market data table
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

            # Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
                ON market_data_enhanced(symbol, timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_cap_sector 
                ON market_data_enhanced(market_cap_category, sector)
            """)

            conn.commit()

            # Verify table creation
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_data_enhanced'")
            if cursor.fetchone():
                logger.info("✅ Database schema created successfully")
            else:
                logger.error("❌ Failed to create database schema")

        except Exception as e:
            logger.error(f"❌ Database schema creation failed: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def test_database_connection(self) -> bool:
        """Test database connection and basic operations"""

        logger.info("🔌 Testing database connection...")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Test basic operations
            cursor.execute("SELECT 1")
            result = cursor.fetchone()

            if result and result[0] == 1:
                logger.info("✅ Database connection successful")

                # Check if tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                logger.info(f"📋 Available tables: {tables}")

                conn.close()
                return True
            else:
                logger.error("❌ Database connection test failed")
                conn.close()
                return False

        except Exception as e:
            logger.error(f"❌ Database connection error: {e}")
            return False

    def populate_training_data(self, days: int = 30, max_stocks: int = 10) -> Tuple[int, int]:
        """Populate database with Indian stock training data with verified persistence"""

        logger.info(f"🚀 Starting data population for up to {max_stocks} Indian stocks...")

        # Create database schema first
        self.create_training_database_schema()

        # Test database connection
        if not self.test_database_connection():
            logger.error("❌ Database connection failed - aborting data population")
            return 0, 0

        # Get stock list
        stocks = []
        for market_cap, sectors in self.indian_universe.items():
            for sector, stock_list in sectors.items():
                stocks.extend(stock_list)

        # Limit to manageable number for testing
        stocks = stocks[:max_stocks]

        successful_stocks = 0
        failed_stocks = 0
        total_records_inserted = 0

        # Process each stock
        for i, symbol in enumerate(stocks):
            try:
                logger.info(f"📈 Processing {symbol} ({i + 1}/{len(stocks)})")

                # Get market cap and sector
                market_cap, sector = self._get_stock_category(symbol)

                # Fetch historical data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)

                # Use yfinance to fetch data
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)

                if not hist.empty:
                    records_inserted = self._insert_stock_data(symbol, hist, market_cap, sector)

                    if records_inserted > 0:
                        successful_stocks += 1
                        total_records_inserted += records_inserted
                        logger.info(f"✅ Successfully added {records_inserted} records for {symbol}")
                    else:
                        failed_stocks += 1
                        logger.warning(f"⚠️ No records inserted for {symbol}")
                else:
                    failed_stocks += 1
                    logger.warning(f"⚠️ No data available for {symbol}")

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                failed_stocks += 1
                logger.error(f"❌ Failed to process {symbol}: {e}")
                continue

        # Verify data persistence
        final_count = self._verify_data_persistence()

        logger.info(f"""
        📊 DATA POPULATION COMPLETE:
        ✅ Successful: {successful_stocks} stocks
        ❌ Failed: {failed_stocks} stocks
        📈 Records inserted: {total_records_inserted} 
        🔍 Records verified: {final_count}
        """)

        return successful_stocks, failed_stocks

    def _insert_stock_data(self, symbol: str, hist: pd.DataFrame, market_cap: str, sector: str) -> int:
        """Insert stock data with transaction safety and verification"""

        inserted_count = 0

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for date, row in hist.iterrows():
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO market_data_enhanced 
                        (symbol, symbol_clean, market_cap_category, sector, timestamp, 
                         open_price, high_price, low_price, close_price, volume, adj_close)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        symbol.replace('.NS', ''),
                        market_cap,
                        sector,
                        date.strftime('%Y-%m-%d'),
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        int(row['Volume']),
                        float(row['Close'])  # Use Close as Adj Close fallback
                    ))
                    inserted_count += 1

                except Exception as e:
                    logger.warning(f"⚠️ Failed to insert record for {symbol} on {date}: {e}")
                    continue

            # Commit transaction
            conn.commit()

            # Verify insertion
            cursor.execute("SELECT COUNT(*) FROM market_data_enhanced WHERE symbol = ?", (symbol,))
            verified_count = cursor.fetchone()[0]

            if verified_count != inserted_count:
                logger.warning(f"⚠️ Insert/verify mismatch for {symbol}: {inserted_count} vs {verified_count}")

            conn.close()
            return inserted_count

        except Exception as e:
            logger.error(f"❌ Database insertion error for {symbol}: {e}")
            return 0

    def _verify_data_persistence(self) -> int:
        """Verify that data was actually persisted to database"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM market_data_enhanced")
            total_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM market_data_enhanced")
            unique_symbols = cursor.fetchone()[0]

            logger.info(f"🔍 Verification: {total_count} total records, {unique_symbols} unique symbols")

            conn.close()
            return total_count

        except Exception as e:
            logger.error(f"❌ Data verification failed: {e}")
            return 0

    def _get_stock_category(self, symbol: str) -> Tuple[str, str]:
        """Get market cap category and sector for a stock"""

        for market_cap, sectors in self.indian_universe.items():
            for sector, stocks in sectors.items():
                if symbol in stocks:
                    return market_cap, sector

        return 'unknown', 'unknown'

    def get_training_statistics(self) -> Dict:
        """Get statistics about the training data with proper error handling"""

        logger.info(f"📊 Reading statistics from: {self.db_path}")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            stats = {}

            # Check if table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='market_data_enhanced'
            """)

            if not cursor.fetchone():
                logger.warning("⚠️ market_data_enhanced table does not exist")
                conn.close()
                return {'error': 'No data table found', 'database_path': self.db_path}

            # Total stocks and records
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM market_data_enhanced")
            stats['total_stocks'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM market_data_enhanced")
            stats['total_records'] = cursor.fetchone()[0]

            # Market cap distribution
            cursor.execute("""
                SELECT market_cap_category, COUNT(DISTINCT symbol) 
                FROM market_data_enhanced 
                GROUP BY market_cap_category
            """)
            stats['market_cap_distribution'] = dict(cursor.fetchall())

            # Sector distribution
            cursor.execute("""
                SELECT sector, COUNT(DISTINCT symbol) 
                FROM market_data_enhanced 
                GROUP BY sector
            """)
            stats['sector_distribution'] = dict(cursor.fetchall())

            # Date range
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM market_data_enhanced")
            date_range = cursor.fetchone()
            stats['date_range'] = {
                'start': date_range[0],
                'end': date_range[1]
            }

            # Add database path for debugging
            stats['database_path'] = self.db_path

            conn.close()
            return stats

        except Exception as e:
            logger.error(f"❌ Statistics retrieval failed: {e}")
            return {'error': str(e), 'database_path': self.db_path}


def main():
    """Main function with comprehensive testing and verification"""
    print("🇮🇳 FIXED INDIAN STOCK DATA POPULATION")
    print("=" * 50)

    # Create fetcher with explicit path
    fetcher = FixedIndianStockDataFetcher()

    print(f"📁 Database location: {fetcher.db_path}")

    # Test database connection first
    if not fetcher.test_database_connection():
        print("❌ Database connection failed - check database setup")
        return

    print("\n🔄 POPULATING DATABASE WITH INDIAN MARKET DATA...")
    successful, failed = fetcher.populate_training_data(days=30, max_stocks=10)

    # Show statistics with error handling
    print("\n📈 TRAINING DATA STATISTICS:")
    stats = fetcher.get_training_statistics()

    if 'error' not in stats:
        print(f"Total Stocks: {stats['total_stocks']}")
        print(f"Total Records: {stats['total_records']}")
        print(f"Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        print(f"Market Cap Distribution: {stats['market_cap_distribution']}")
        print(f"Sector Distribution: {stats['sector_distribution']}")
        print(f"Database Path: {stats['database_path']}")
    else:
        print(f"❌ Error retrieving statistics: {stats['error']}")
        print(f"Database Path: {stats.get('database_path', 'Unknown')}")

    print("\n🎯 FIXED DATA FETCHING COMPLETE!")


if __name__ == "__main__":
    main()
'''

        # Write the corrected fetcher
        corrected_path = os.path.join(self.base_path, "06_DATA", "indian_stock_data_fetcher_corrected.py")
        os.makedirs(os.path.dirname(corrected_path), exist_ok=True)

        with open(corrected_path, 'w') as f:
            f.write(fixed_fetcher_code)

        logger.info(f"✅ Created corrected data fetcher: {corrected_path}")
        return corrected_path


def main():
    """Main diagnosis and fix process"""
    print("🔧 DATABASE PATH DIAGNOSIS AND FIX")
    print("=" * 50)

    validator = DatabasePathValidator()

    # Step 1: Diagnose current issues
    print("\n🔍 Step 1: Diagnosing database path issues...")
    diagnosis = validator.diagnose_database_issue()

    print(f"📁 Found {len(diagnosis['database_files_found'])} database files:")
    for db_path in diagnosis['database_files_found']:
        count = diagnosis['record_counts'].get(db_path, 0)
        if isinstance(count, int) and count > 0:
            print(f"  ✅ {db_path}: {count} Indian stock records")
        else:
            print(f"  ⚪ {db_path}: {count}")

    print(f"\n🚨 Issues identified: {len(diagnosis['issue_summary'])}")
    for issue in diagnosis['issue_summary']:
        print(f"  - {issue}")

    # Step 2: Fix database paths
    print("\n🔧 Step 2: Fixing database path configuration...")
    success = validator.fix_database_paths()

    if success:
        print("✅ Database paths fixed successfully")
    else:
        print("❌ Failed to fix database paths")
        return

    # Step 3: Create corrected data fetcher
    print("\n📝 Step 3: Creating corrected data fetcher...")
    corrected_path = validator.create_fixed_data_fetcher()

    print(f"✅ Corrected data fetcher created at: {corrected_path}")

    print("\n🎯 NEXT STEPS:")
    print("1. Run the corrected data fetcher:")
    print("   python 06_DATA/indian_stock_data_fetcher_corrected.py")
    print("2. Verify data persistence and statistics")
    print("3. Continue with database consolidation if needed")


if __name__ == "__main__":
    main()