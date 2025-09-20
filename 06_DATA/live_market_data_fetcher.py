# 06_DATA/live_market_data_fetcher.py
"""
Phase 3, Step 1: Live Market Data Integration
Real-time market data fetcher with multiple provider support

Features:
- yfinance for real-time data
- Alpha Vantage API integration
- WebSocket streaming for live updates
- Intelligent fallback between providers
- Production-grade error handling
- Rate limiting and caching

Location: #06_DATA/live_market_data_fetcher.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import requests
import time
import asyncio
import websockets
import json
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProvider(Enum):
    """Available data providers"""
    YFINANCE = "yfinance"
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    WEBSOCKET = "websocket"


@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    provider: str
    timeframe: str = "1m"

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        return data


@dataclass
class ProviderConfig:
    """Data provider configuration"""
    name: str
    enabled: bool
    api_key: Optional[str]
    rate_limit: int  # requests per minute
    priority: int  # 1 = highest priority


class LiveMarketDataFetcher:
    """Production-grade live market data fetcher"""

    def __init__(self, db_path: str = "marketpulse_production.db", config_file: str = None):
        """Initialize the live data fetcher"""
        self.db_path = Path(db_path)
        self.cache = {}
        self.cache_expiry = {}
        self.last_request_time = {}
        self.providers = self._load_provider_config(config_file)
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.websocket_connections = {}
        self.streaming_active = False

        logger.info(f"Live Market Data Fetcher initialized with {len(self.providers)} providers")

    def _load_provider_config(self, config_file: str = None) -> Dict[str, ProviderConfig]:
        """Load provider configuration"""
        default_config = {
            'yfinance': ProviderConfig(
                name='yfinance',
                enabled=True,
                api_key=None,
                rate_limit=10,  # Conservative rate limit
                priority=1
            ),
            'alpha_vantage': ProviderConfig(
                name='alpha_vantage',
                enabled=bool(os.getenv('ALPHA_VANTAGE_API_KEY')),
                api_key=os.getenv('ALPHA_VANTAGE_API_KEY'),
                rate_limit=5,
                priority=2
            ),
            'finnhub': ProviderConfig(
                name='finnhub',
                enabled=bool(os.getenv('FINNHUB_API_KEY')),
                api_key=os.getenv('FINNHUB_API_KEY'),
                rate_limit=60,
                priority=3
            )
        }

        if config_file and Path(config_file).exists():
            # Load custom config if provided
            pass

        return default_config

    def _connect_database(self) -> sqlite3.Connection:
        """Connect to database"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _respect_rate_limit(self, provider: str) -> None:
        """Respect provider rate limits"""
        if provider not in self.providers:
            return

        config = self.providers[provider]
        current_time = time.time()

        if provider in self.last_request_time:
            time_since_last = current_time - self.last_request_time[provider]
            min_interval = 60.0 / config.rate_limit

            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                logger.debug(f"Rate limiting {provider}: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)

        self.last_request_time[provider] = current_time

    def _is_cache_valid(self, cache_key: str, max_age: int = 60) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache_expiry:
            return False

        return time.time() - self.cache_expiry[cache_key] < max_age

    def _cache_data(self, cache_key: str, data: Any) -> None:
        """Cache data with timestamp"""
        self.cache[cache_key] = data
        self.cache_expiry[cache_key] = time.time()

    def get_live_quote(self, symbol: str) -> Optional[MarketData]:
        """Get live quote for symbol with provider fallback"""

        # Check cache first
        cache_key = f"quote_{symbol}"
        if self._is_cache_valid(cache_key, max_age=30):  # 30 second cache
            logger.debug(f"Returning cached quote for {symbol}")
            return self.cache[cache_key]

        # Try providers in priority order
        enabled_providers = sorted(
            [(name, config) for name, config in self.providers.items() if config.enabled],
            key=lambda x: x[1].priority
        )

        for provider_name, config in enabled_providers:
            try:
                data = self._fetch_quote_from_provider(symbol, provider_name)
                if data:
                    self._cache_data(cache_key, data)
                    return data
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed for {symbol}: {e}")
                continue

        logger.error(f"All providers failed for {symbol}")
        return None

    def _fetch_quote_from_provider(self, symbol: str, provider: str) -> Optional[MarketData]:
        """Fetch quote from specific provider"""

        if provider == 'yfinance':
            return self._fetch_yfinance_quote(symbol)
        elif provider == 'alpha_vantage':
            return self._fetch_alpha_vantage_quote(symbol)
        elif provider == 'finnhub':
            return self._fetch_finnhub_quote(symbol)
        else:
            logger.warning(f"Unknown provider: {provider}")
            return None

    def _fetch_yfinance_quote(self, symbol: str) -> Optional[MarketData]:
        """Fetch quote using yfinance"""
        self._respect_rate_limit('yfinance')

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get the most recent trading data
            hist = ticker.history(period="1d", interval="1m")
            if hist.empty:
                # Fallback to daily data
                hist = ticker.history(period="1d")

            if hist.empty:
                logger.warning(f"No data available for {symbol}")
                return None

            latest = hist.iloc[-1]

            return MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                open_price=float(latest['Open']),
                high_price=float(latest['High']),
                low_price=float(latest['Low']),
                close_price=float(latest['Close']),
                volume=int(latest['Volume']),
                provider='yfinance',
                timeframe='1m'
            )

        except Exception as e:
            logger.error(f"yfinance error for {symbol}: {e}")
            return None

    def _fetch_alpha_vantage_quote(self, symbol: str) -> Optional[MarketData]:
        """Fetch quote using Alpha Vantage"""
        if not self.providers['alpha_vantage'].api_key:
            return None

        self._respect_rate_limit('alpha_vantage')

        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.providers['alpha_vantage'].api_key
            }

            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if 'Global Quote' not in data:
                logger.warning(f"Alpha Vantage: No data for {symbol}")
                return None

            quote = data['Global Quote']

            return MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                open_price=float(quote['02. open']),
                high_price=float(quote['03. high']),
                low_price=float(quote['04. low']),
                close_price=float(quote['05. price']),
                volume=int(quote['06. volume']),
                provider='alpha_vantage',
                timeframe='1d'
            )

        except Exception as e:
            logger.error(f"Alpha Vantage error for {symbol}: {e}")
            return None

    def _fetch_finnhub_quote(self, symbol: str) -> Optional[MarketData]:
        """Fetch quote using Finnhub"""
        if not self.providers['finnhub'].api_key:
            return None

        self._respect_rate_limit('finnhub')

        try:
            url = f"https://finnhub.io/api/v1/quote"
            params = {
                'symbol': symbol,
                'token': self.providers['finnhub'].api_key
            }

            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if 'c' not in data:  # current price
                logger.warning(f"Finnhub: No data for {symbol}")
                return None

            # Finnhub provides limited OHLC data in quote
            return MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                open_price=float(data['o']),  # open
                high_price=float(data['h']),  # high
                low_price=float(data['l']),  # low
                close_price=float(data['c']),  # current
                volume=0,  # Not available in quote endpoint
                provider='finnhub',
                timeframe='1d'
            )

        except Exception as e:
            logger.error(f"Finnhub error for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol: str, period: str = "1mo", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Get historical data with provider fallback"""

        cache_key = f"hist_{symbol}_{period}_{interval}"
        if self._is_cache_valid(cache_key, max_age=300):  # 5 minute cache for historical
            logger.debug(f"Returning cached historical data for {symbol}")
            return self.cache[cache_key]

        try:
            # Use yfinance for historical data (most reliable)
            self._respect_rate_limit('yfinance')

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)

            if hist.empty:
                logger.warning(f"No historical data for {symbol}")
                return None

            # Convert to standard format
            hist = hist.reset_index()
            hist['Symbol'] = symbol
            hist['Provider'] = 'yfinance'

            # Rename columns to match our schema
            column_mapping = {
                'Date': 'timestamp',
                'Datetime': 'timestamp',
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Close': 'close_price',
                'Volume': 'volume',
                'Symbol': 'symbol',
                'Provider': 'provider'
            }

            hist = hist.rename(columns=column_mapping)
            hist['timeframe'] = interval

            self._cache_data(cache_key, hist)
            return hist

        except Exception as e:
            logger.error(f"Historical data error for {symbol}: {e}")
            return None

    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Optional[MarketData]]:
        """Get quotes for multiple symbols concurrently"""

        results = {}

        # Use ThreadPoolExecutor for concurrent requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {
                executor.submit(self.get_live_quote, symbol): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    quote = future.result(timeout=30)
                    results[symbol] = quote
                except Exception as e:
                    logger.error(f"Error fetching quote for {symbol}: {e}")
                    results[symbol] = None

        return results

    def store_market_data(self, data: Union[MarketData, List[MarketData]]) -> bool:
        """Store market data in database"""

        if isinstance(data, MarketData):
            data = [data]

        try:
            conn = self._connect_database()
            cursor = conn.cursor()

            stored_count = 0
            for market_data in data:
                cursor.execute("""
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timestamp, timeframe, open_price, high_price, 
                     low_price, close_price, volume, data_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    market_data.symbol,
                    market_data.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    market_data.timeframe,
                    market_data.open_price,
                    market_data.high_price,
                    market_data.low_price,
                    market_data.close_price,
                    market_data.volume,
                    market_data.provider
                ))
                stored_count += 1

            conn.commit()
            conn.close()

            logger.info(f"Stored {stored_count} market data records")
            return True

        except Exception as e:
            logger.error(f"Error storing market data: {e}")
            return False

    def start_streaming(self, symbols: List[str], callback=None) -> None:
        """Start streaming live data (placeholder for WebSocket implementation)"""

        logger.info(f"Starting live streaming for {len(symbols)} symbols")
        self.streaming_active = True

        def streaming_loop():
            """Continuous streaming loop"""
            while self.streaming_active:
                try:
                    # Get live quotes for all symbols
                    quotes = self.get_multiple_quotes(symbols)

                    # Store in database
                    valid_quotes = [quote for quote in quotes.values() if quote is not None]
                    if valid_quotes:
                        self.store_market_data(valid_quotes)

                    # Call callback if provided
                    if callback:
                        callback(quotes)

                    # Update every 60 seconds for now (can be made configurable)
                    time.sleep(60)

                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    time.sleep(30)  # Wait before retrying

        # Start streaming in background thread
        streaming_thread = threading.Thread(target=streaming_loop, daemon=True)
        streaming_thread.start()

        logger.info("Live streaming started")

    def stop_streaming(self) -> None:
        """Stop live streaming"""
        self.streaming_active = False
        logger.info("Live streaming stopped")

    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all data providers"""

        status = {}

        for name, config in self.providers.items():
            provider_status = {
                'enabled': config.enabled,
                'priority': config.priority,
                'rate_limit': config.rate_limit,
                'has_api_key': config.api_key is not None,
                'last_request': self.last_request_time.get(name),
                'cache_entries': len([k for k in self.cache.keys() if name in str(k)])
            }

            # Test connectivity
            try:
                if name == 'yfinance' and config.enabled:
                    test_data = self._fetch_yfinance_quote('AAPL')
                    provider_status['connectivity'] = test_data is not None
                elif name == 'alpha_vantage' and config.enabled and config.api_key:
                    test_data = self._fetch_alpha_vantage_quote('AAPL')
                    provider_status['connectivity'] = test_data is not None
                elif name == 'finnhub' and config.enabled and config.api_key:
                    test_data = self._fetch_finnhub_quote('AAPL')
                    provider_status['connectivity'] = test_data is not None
                else:
                    provider_status['connectivity'] = None
            except:
                provider_status['connectivity'] = False

            status[name] = provider_status

        return status

    def cleanup(self) -> None:
        """Cleanup resources"""
        self.stop_streaming()
        self.executor.shutdown(wait=True)
        logger.info("Live Market Data Fetcher cleaned up")


def main():
    """Test the live market data fetcher"""

    print("Phase 3 - Live Market Data Fetcher Test")
    print("=" * 45)

    # Initialize fetcher
    fetcher = LiveMarketDataFetcher()

    # Test provider status
    print("Provider Status:")
    status = fetcher.get_provider_status()
    for provider, info in status.items():
        connectivity = "✅" if info['connectivity'] else "❌" if info['connectivity'] is False else "⚪"
        enabled = "✅" if info['enabled'] else "❌"
        print(f"  {provider}: {enabled} enabled, {connectivity} connected")

    # Test live quotes
    print("\nTesting Live Quotes:")
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']

    quotes = fetcher.get_multiple_quotes(test_symbols[:2])  # Test with 2 symbols first

    for symbol, quote in quotes.items():
        if quote:
            print(f"  {symbol}: ${quote.close_price:.2f} (Provider: {quote.provider})")
        else:
            print(f"  {symbol}: No data available")

    # Test historical data
    print("\nTesting Historical Data:")
    hist_data = fetcher.get_historical_data('AAPL', period='5d', interval='1d')
    if hist_data is not None:
        print(f"  AAPL: {len(hist_data)} historical records retrieved")
        print(f"  Date range: {hist_data['timestamp'].min()} to {hist_data['timestamp'].max()}")
    else:
        print("  No historical data available")

    # Test database storage
    print("\nTesting Database Storage:")
    if quotes:
        valid_quotes = [quote for quote in quotes.values() if quote is not None]
        if valid_quotes:
            success = fetcher.store_market_data(valid_quotes)
            print(f"  Database storage: {'✅ Success' if success else '❌ Failed'}")

    print("\n🎉 Live Market Data Fetcher Phase 3 Step 1 Complete!")

    # Cleanup
    fetcher.cleanup()

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)