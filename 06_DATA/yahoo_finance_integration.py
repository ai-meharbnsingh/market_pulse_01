# 06_DATA/yahoo_finance_integration.py
"""
Production Yahoo Finance Integration - Master Document Priority #1
Real-time market data fetcher for production deployment

Features:
- Real-time market data from Yahoo Finance
- Historical data fetching with multiple timeframes  
- Error handling and rate limiting
- Caching for performance optimization
- Multi-symbol support for portfolio tracking
- Data validation and cleaning
- Connection health monitoring
- Fallback mechanisms for reliability

Location: #06_DATA/yahoo_finance_integration.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import logging
import time
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from functools import lru_cache
import sys
import os

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YahooFinanceError(Exception):
    """Custom exception for Yahoo Finance related errors"""
    pass


class ConnectionHealthMonitor:
    """Monitor Yahoo Finance API connection health"""

    def __init__(self):
        self.last_successful_call = None
        self.failed_calls = 0
        self.total_calls = 0
        self.average_response_time = 0.0
        self.lock = threading.Lock()

    def record_call(self, success: bool, response_time: float):
        """Record API call result and timing"""
        with self.lock:
            self.total_calls += 1
            if success:
                self.last_successful_call = datetime.now()
                self.failed_calls = max(0, self.failed_calls - 1)  # Decay failed calls
            else:
                self.failed_calls += 1

            # Update average response time (exponential moving average)
            alpha = 0.1
            self.average_response_time = (alpha * response_time +
                                          (1 - alpha) * self.average_response_time)

    def get_health_status(self) -> Dict[str, Any]:
        """Get current connection health status"""
        with self.lock:
            if self.total_calls == 0:
                return {"status": "unknown", "reason": "no_calls_made"}

            failure_rate = self.failed_calls / max(1, self.total_calls)
            time_since_success = None

            if self.last_successful_call:
                time_since_success = (datetime.now() - self.last_successful_call).total_seconds()

            if failure_rate > 0.5:
                status = "poor"
            elif failure_rate > 0.2:
                status = "degraded"
            elif time_since_success and time_since_success > 300:  # 5 minutes
                status = "stale"
            else:
                status = "healthy"

            return {
                "status": status,
                "failure_rate": failure_rate,
                "failed_calls": self.failed_calls,
                "total_calls": self.total_calls,
                "avg_response_time": self.average_response_time,
                "last_successful_call": self.last_successful_call,
                "time_since_success": time_since_success
            }


class DataCache:
    """Simple in-memory cache for market data"""

    def __init__(self, default_ttl: int = 60):
        self.cache = {}
        self.timestamps = {}
        self.default_ttl = default_ttl
        self.lock = threading.Lock()

    def get(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """Get cached data if still valid"""
        ttl = ttl or self.default_ttl
        with self.lock:
            if key in self.cache:
                age = time.time() - self.timestamps[key]
                if age < ttl:
                    return self.cache[key]
                else:
                    # Expired, remove from cache
                    del self.cache[key]
                    del self.timestamps[key]
            return None

    def set(self, key: str, value: Any):
        """Store data in cache"""
        with self.lock:
            self.cache[key] = value
            self.timestamps[key] = time.time()

    def clear(self):
        """Clear all cached data"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                "entries": len(self.cache),
                "oldest_entry": min(self.timestamps.values()) if self.timestamps else None,
                "newest_entry": max(self.timestamps.values()) if self.timestamps else None
            }


class YahooFinanceIntegration:
    """Production-ready Yahoo Finance integration for real market data"""

    def __init__(self, db_path: str = "marketpulse_production.db", cache_ttl: int = 30):
        """
        Initialize Yahoo Finance integration

        Args:
            db_path: Path to SQLite database
            cache_ttl: Cache time-to-live in seconds (default 30s for real-time)
        """
        self.db_path = db_path
        self.cache = DataCache(default_ttl=cache_ttl)
        self.health_monitor = ConnectionHealthMonitor()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MarketPulse/1.0 (Trading System)'
        })

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        self.lock = threading.Lock()

        # Database connection management for Windows compatibility
        self._db_connections = []
        self._connection_lock = threading.Lock()

        # Initialize database
        self._init_database()

        logger.info("Yahoo Finance Integration initialized for production")

    def _get_db_connection(self):
        """Get database connection with Windows-compatible management"""
        with self._connection_lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            # Set pragmas for better Windows compatibility
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            self._db_connections.append(conn)
            return conn

    def _close_db_connection(self, conn):
        """Properly close database connection"""
        try:
            if conn:
                conn.close()
                with self._connection_lock:
                    if conn in self._db_connections:
                        self._db_connections.remove(conn)
        except Exception as e:
            logger.warning(f"Error closing database connection: {e}")

    def close_all_connections(self):
        """Close all database connections - call before cleanup"""
        with self._connection_lock:
            connections_to_close = self._db_connections.copy()
            self._db_connections.clear()

        for conn in connections_to_close:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection during cleanup: {e}")

        # Force garbage collection to release resources
        import gc
        gc.collect()

        # Small delay for Windows file system
        import time
        time.sleep(0.1)

    def _init_database(self):
        """Initialize SQLite database with market data tables"""
        conn = None
        try:
            conn = self._get_db_connection()

            # Market data table with proper schema
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data_live (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    adj_close REAL,
                    source TEXT DEFAULT 'yahoo_finance',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, source)
                )
            """)

            # Real-time quotes table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS real_time_quotes (
                    symbol TEXT PRIMARY KEY,
                    current_price REAL NOT NULL,
                    bid REAL,
                    ask REAL,
                    bid_size INTEGER,
                    ask_size INTEGER,
                    day_change REAL,
                    day_change_percent REAL,
                    day_high REAL,
                    day_low REAL,
                    day_volume INTEGER,
                    market_cap REAL,
                    pe_ratio REAL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Connection health log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS connection_health_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL,
                    failure_rate REAL,
                    response_time REAL,
                    details TEXT
                )
            """)

            # Create indexes for performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data_live(symbol, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quotes_symbol ON real_time_quotes(symbol)")

            conn.commit()
            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            # Don't raise here - let the system continue with fallback behavior
            print(f"❌ Database initialization error: {e}")
        finally:
            if conn:
                self._close_db_connection(conn)

    def _rate_limit(self):
        """Implement rate limiting for Yahoo Finance requests"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                time.sleep(sleep_time)

            self.last_request_time = time.time()

    def get_real_time_quote(self, symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get real-time quote for a symbol

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
            use_cache: Whether to use cached data if available

        Returns:
            Dictionary with quote data or None if failed
        """
        cache_key = f"quote_{symbol}"

        # Check cache first
        if use_cache:
            cached_data = self.cache.get(cache_key, ttl=30)  # 30 second cache for quotes
            if cached_data:
                return cached_data

        start_time = time.time()
        success = False

        try:
            self._rate_limit()

            # Fetch from Yahoo Finance
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get current price and other real-time data
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price:
                logger.warning(f"No current price found for {symbol}")
                return None

            quote_data = {
                'symbol': symbol,
                'current_price': current_price,
                'bid': info.get('bid'),
                'ask': info.get('ask'),
                'bid_size': info.get('bidSize'),
                'ask_size': info.get('askSize'),
                'day_change': info.get('regularMarketChange'),
                'day_change_percent': info.get('regularMarketChangePercent'),
                'day_high': info.get('dayHigh') or info.get('regularMarketDayHigh'),
                'day_low': info.get('dayLow') or info.get('regularMarketDayLow'),
                'day_volume': info.get('volume') or info.get('regularMarketVolume'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('forwardPE') or info.get('trailingPE'),
                'updated_at': datetime.now()
            }

            # Cache the result
            self.cache.set(cache_key, quote_data)

            # Store in database
            self._store_real_time_quote(quote_data)

            success = True
            logger.debug(f"Successfully fetched quote for {symbol}: ${current_price}")
            return quote_data

        except Exception as e:
            logger.error(f"Failed to fetch quote for {symbol}: {e}")
            return None

        finally:
            response_time = time.time() - start_time
            self.health_monitor.record_call(success, response_time)

    def get_historical_data(self, symbol: str, period: str = "1mo",
                            interval: str = "1d", use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Get historical market data for a symbol

        Args:
            symbol: Stock symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with historical data or None if failed
        """
        cache_key = f"history_{symbol}_{period}_{interval}"

        # Check cache first (longer TTL for historical data)
        if use_cache:
            cached_data = self.cache.get(cache_key, ttl=300)  # 5 minute cache
            if cached_data is not None:
                return cached_data

        start_time = time.time()
        success = False

        try:
            self._rate_limit()

            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=period, interval=interval)

            if hist_data.empty:
                logger.warning(f"No historical data found for {symbol}")
                return None

            # Clean and validate data
            hist_data = hist_data.dropna()

            # Store in database
            self._store_historical_data(symbol, hist_data)

            # Cache the result
            self.cache.set(cache_key, hist_data)

            success = True
            logger.debug(f"Successfully fetched {len(hist_data)} historical records for {symbol}")
            return hist_data

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return None

        finally:
            response_time = time.time() - start_time
            self.health_monitor.record_call(success, response_time)

    def get_multiple_quotes(self, symbols: List[str], use_cache: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Get real-time quotes for multiple symbols efficiently

        Args:
            symbols: List of stock symbols
            use_cache: Whether to use cached data

        Returns:
            Dictionary mapping symbols to quote data
        """
        results = {}

        # Use ThreadPoolExecutor for concurrent requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {
                executor.submit(self.get_real_time_quote, symbol, use_cache): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    quote_data = future.result()
                    if quote_data:
                        results[symbol] = quote_data
                except Exception as e:
                    logger.error(f"Failed to fetch quote for {symbol}: {e}")

        logger.info(f"Successfully fetched {len(results)}/{len(symbols)} quotes")
        return results

    def _store_real_time_quote(self, quote_data: Dict[str, Any]):
        """Store real-time quote in database"""
        conn = None
        try:
            conn = self._get_db_connection()
            conn.execute("""
                INSERT OR REPLACE INTO real_time_quotes 
                (symbol, current_price, bid, ask, bid_size, ask_size, 
                 day_change, day_change_percent, day_high, day_low, 
                 day_volume, market_cap, pe_ratio, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                quote_data['symbol'], quote_data['current_price'],
                quote_data['bid'], quote_data['ask'],
                quote_data['bid_size'], quote_data['ask_size'],
                quote_data['day_change'], quote_data['day_change_percent'],
                quote_data['day_high'], quote_data['day_low'],
                quote_data['day_volume'], quote_data['market_cap'],
                quote_data['pe_ratio'], quote_data['updated_at']
            ))
            conn.commit()

        except Exception as e:
            logger.error(f"Failed to store quote data: {e}")
        finally:
            if conn:
                self._close_db_connection(conn)

    def _store_historical_data(self, symbol: str, hist_data: pd.DataFrame):
        """Store historical data in database"""
        conn = None
        try:
            conn = self._get_db_connection()
            for timestamp, row in hist_data.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO market_data_live 
                    (symbol, timestamp, open_price, high_price, low_price, 
                     close_price, volume, adj_close, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, timestamp, row['Open'], row['High'], row['Low'],
                    row['Close'], row['Volume'], row.get('Adj Close', row['Close']),
                    'yahoo_finance'
                ))
            conn.commit()

        except Exception as e:
            logger.error(f"Failed to store historical data: {e}")
        finally:
            if conn:
                self._close_db_connection(conn)

    def get_connection_health(self) -> Dict[str, Any]:
        """Get current Yahoo Finance connection health status"""
        health_status = self.health_monitor.get_health_status()
        cache_stats = self.cache.get_stats()

        # Log health status to database
        conn = None
        try:
            conn = self._get_db_connection()
            # Prepare serializable health data
            health_data_for_json = {}
            for key, value in {**health_status, **cache_stats}.items():
                if isinstance(value, datetime):
                    health_data_for_json[key] = value.isoformat() if value else None
                else:
                    health_data_for_json[key] = value

            conn.execute("""
                INSERT INTO connection_health_log 
                (status, failure_rate, response_time, details)
                VALUES (?, ?, ?, ?)
            """, (
                health_status['status'],
                health_status.get('failure_rate', 0),
                health_status.get('avg_response_time', 0),
                json.dumps(health_data_for_json)
            ))
            conn.commit()

        except Exception as e:
            logger.error(f"Failed to log health status: {e}")
        finally:
            if conn:
                self._close_db_connection(conn)

        return {
            'yahoo_finance_health': health_status,
            'cache_stats': cache_stats,
            'database_path': self.db_path
        }

    def test_connection(self) -> bool:
        """Test Yahoo Finance connection with a simple request"""
        try:
            test_quote = self.get_real_time_quote("AAPL", use_cache=False)
            return test_quote is not None
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("Cache cleared")

    def get_portfolio_quotes(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get portfolio quotes as a formatted DataFrame

        Args:
            symbols: List of portfolio symbols

        Returns:
            DataFrame with portfolio quote data
        """
        quotes = self.get_multiple_quotes(symbols)

        if not quotes:
            return pd.DataFrame()

        # Convert to DataFrame for easy manipulation
        df_data = []
        for symbol, quote in quotes.items():
            df_data.append({
                'Symbol': symbol,
                'Price': quote['current_price'],
                'Change': quote.get('day_change', 0),
                'Change%': quote.get('day_change_percent', 0),
                'Volume': quote.get('day_volume', 0),
                'Bid': quote.get('bid'),
                'Ask': quote.get('ask'),
                'High': quote.get('day_high'),
                'Low': quote.get('day_low'),
                'Market Cap': quote.get('market_cap'),
                'P/E': quote.get('pe_ratio'),
                'Updated': quote['updated_at']
            })

        return pd.DataFrame(df_data)


def test_yahoo_integration():
    """Test function for Yahoo Finance integration"""
    print("Testing Yahoo Finance Integration...")

    # Initialize integration
    yf_integration = YahooFinanceIntegration()

    # Test connection
    print(f"Connection test: {'PASSED' if yf_integration.test_connection() else 'FAILED'}")

    # Test single quote
    quote = yf_integration.get_real_time_quote("AAPL")
    if quote:
        print(f"AAPL Quote: ${quote['current_price']:.2f}")

    # Test multiple quotes
    portfolio_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    quotes = yf_integration.get_multiple_quotes(portfolio_symbols)
    print(f"Portfolio quotes fetched: {len(quotes)}/{len(portfolio_symbols)}")

    # Test historical data
    hist_data = yf_integration.get_historical_data("AAPL", period="1mo")
    if hist_data is not None:
        print(f"Historical data: {len(hist_data)} records")

    # Check health
    health = yf_integration.get_connection_health()
    print(f"Health Status: {health['yahoo_finance_health']['status']}")

    return yf_integration


if __name__ == "__main__":
    # Run test when executed directly
    integration = test_yahoo_integration()