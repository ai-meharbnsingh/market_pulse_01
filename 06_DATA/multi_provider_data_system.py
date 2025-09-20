# 06_DATA/multi_provider_data_system.py
"""
Multi-Provider Fallback Data System
Reliable market data with Yahoo Finance, Alpha Vantage, and Finnhub

Features:
- Intelligent provider failover system
- Rate limiting per provider
- Data quality validation
- Provider health monitoring
- Unified data format across providers
- Real-time quote aggregation
- Historical data fallback chain
- Provider ranking based on performance

Location: #06_DATA/multi_provider_data_system.py
"""

import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import logging
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import sys

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProvider(Enum):
    """Supported data providers"""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    DEMO = "demo"


class ProviderStatus(Enum):
    """Provider health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    UNKNOWN = "unknown"


class ProviderHealthMonitor:
    """Monitor individual provider health and performance"""

    def __init__(self, provider: DataProvider):
        self.provider = provider
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_calls = 0
        self.last_successful_call = None
        self.last_failed_call = None
        self.average_response_time = 0.0
        self.current_status = ProviderStatus.UNKNOWN
        self.rate_limit_reset_time = None
        self.lock = threading.Lock()

    def record_call(self, success: bool, response_time: float, error_type: str = None):
        """Record API call result"""
        with self.lock:
            self.total_calls += 1

            if success:
                self.successful_calls += 1
                self.last_successful_call = datetime.now()
                if self.current_status == ProviderStatus.FAILED:
                    self.current_status = ProviderStatus.DEGRADED
                elif self.get_success_rate() > 0.8:
                    self.current_status = ProviderStatus.HEALTHY
            else:
                self.failed_calls += 1
                self.last_failed_call = datetime.now()

                # Determine failure type
                if error_type and "429" in error_type:
                    self.current_status = ProviderStatus.RATE_LIMITED
                    self.rate_limit_reset_time = datetime.now() + timedelta(minutes=5)
                elif self.get_success_rate() < 0.2:
                    self.current_status = ProviderStatus.FAILED
                else:
                    self.current_status = ProviderStatus.DEGRADED

            # Update average response time
            alpha = 0.1
            self.average_response_time = (alpha * response_time +
                                          (1 - alpha) * self.average_response_time)

    def get_success_rate(self) -> float:
        """Get current success rate"""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    def is_available(self) -> bool:
        """Check if provider is currently available"""
        if self.current_status == ProviderStatus.RATE_LIMITED:
            if self.rate_limit_reset_time and datetime.now() > self.rate_limit_reset_time:
                self.current_status = ProviderStatus.DEGRADED
                return True
            return False

        return self.current_status != ProviderStatus.FAILED

    def get_health_score(self) -> float:
        """Get overall health score (0-100)"""
        if self.total_calls == 0:
            return 50.0  # Unknown

        success_score = self.get_success_rate() * 100

        # Adjust for response time (prefer faster providers)
        time_score = max(0, 100 - self.average_response_time * 10)

        # Recent activity bonus
        recency_score = 100
        if self.last_successful_call:
            minutes_since = (datetime.now() - self.last_successful_call).total_seconds() / 60
            recency_score = max(0, 100 - minutes_since)

        return (success_score * 0.5 + time_score * 0.3 + recency_score * 0.2)


class YahooFinanceProvider:
    """Yahoo Finance data provider implementation"""

    def __init__(self):
        self.name = DataProvider.YAHOO_FINANCE
        self.health = ProviderHealthMonitor(self.name)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MarketPulse/1.0'
        })
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
        self.lock = threading.Lock()

    def _rate_limit(self):
        """Apply rate limiting"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - time_since_last)
            self.last_request_time = time.time()

    def get_real_time_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote from Yahoo Finance"""
        start_time = time.time()
        success = False

        try:
            self._rate_limit()

            ticker = yf.Ticker(symbol)
            info = ticker.info

            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price:
                return None

            quote_data = {
                'symbol': symbol,
                'price': float(current_price),
                'bid': info.get('bid'),
                'ask': info.get('ask'),
                'volume': info.get('volume') or info.get('regularMarketVolume'),
                'change': info.get('regularMarketChange'),
                'change_percent': info.get('regularMarketChangePercent'),
                'high': info.get('dayHigh') or info.get('regularMarketDayHigh'),
                'low': info.get('dayLow') or info.get('regularMarketDayLow'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('forwardPE') or info.get('trailingPE'),
                'provider': self.name.value,
                'timestamp': datetime.now()
            }

            success = True
            return quote_data

        except Exception as e:
            logger.debug(f"Yahoo Finance quote failed for {symbol}: {e}")
            return None
        finally:
            response_time = time.time() - start_time
            self.health.record_call(success, response_time, str(e) if not success else None)

    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical data from Yahoo Finance"""
        start_time = time.time()
        success = False

        try:
            self._rate_limit()

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")

            if hist.empty:
                return None

            # Standardize column names
            hist = hist.reset_index()
            hist['Symbol'] = symbol
            hist['Provider'] = self.name.value

            success = True
            return hist

        except Exception as e:
            logger.debug(f"Yahoo Finance history failed for {symbol}: {e}")
            return None
        finally:
            response_time = time.time() - start_time
            self.health.record_call(success, response_time, str(e) if not success else None)


class AlphaVantageProvider:
    """Alpha Vantage data provider implementation"""

    def __init__(self, api_key: str = None):
        self.name = DataProvider.ALPHA_VANTAGE
        self.health = ProviderHealthMonitor(self.name)
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        self.base_url = "https://www.alphavantage.co/query"
        self.session = requests.Session()
        self.rate_limit_delay = 12  # 5 calls per minute for free tier
        self.last_request_time = 0
        self.lock = threading.Lock()

    def _rate_limit(self):
        """Apply rate limiting for Alpha Vantage"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - time_since_last)
            self.last_request_time = time.time()

    def get_real_time_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote from Alpha Vantage"""
        if self.api_key == 'demo':
            return None

        start_time = time.time()
        success = False

        try:
            self._rate_limit()

            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }

            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if 'Global Quote' not in data:
                return None

            quote = data['Global Quote']

            quote_data = {
                'symbol': symbol,
                'price': float(quote.get('05. price', 0)),
                'change': float(quote.get('09. change', 0)),
                'change_percent': float(quote.get('10. change percent', '0%').replace('%', '')),
                'high': float(quote.get('03. high', 0)),
                'low': float(quote.get('04. low', 0)),
                'volume': int(quote.get('06. volume', 0)),
                'provider': self.name.value,
                'timestamp': datetime.now()
            }

            success = True
            return quote_data

        except Exception as e:
            logger.debug(f"Alpha Vantage quote failed for {symbol}: {e}")
            return None
        finally:
            response_time = time.time() - start_time
            self.health.record_call(success, response_time, str(e) if not success else None)

    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical data from Alpha Vantage"""
        if self.api_key == 'demo':
            return None

        start_time = time.time()
        success = False

        try:
            self._rate_limit()

            function = 'TIME_SERIES_DAILY' if days <= 100 else 'TIME_SERIES_DAILY_ADJUSTED'
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'compact' if days <= 100 else 'full'
            }

            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            time_series_key = 'Time Series (Daily)'
            if time_series_key not in data:
                return None

            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Standardize column names
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.astype(float)
            df = df.reset_index()
            df.rename(columns={'index': 'Date'}, inplace=True)
            df['Symbol'] = symbol
            df['Provider'] = self.name.value

            # Filter to requested days
            if len(df) > days:
                df = df.tail(days)

            success = True
            return df

        except Exception as e:
            logger.debug(f"Alpha Vantage history failed for {symbol}: {e}")
            return None
        finally:
            response_time = time.time() - start_time
            self.health.record_call(success, response_time, str(e) if not success else None)


class FinnhubProvider:
    """Finnhub data provider implementation"""

    def __init__(self, api_key: str = None):
        self.name = DataProvider.FINNHUB
        self.health = ProviderHealthMonitor(self.name)
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY', 'demo')
        self.base_url = "https://finnhub.io/api/v1"
        self.session = requests.Session()
        self.rate_limit_delay = 1  # 60 calls per minute for free tier
        self.last_request_time = 0
        self.lock = threading.Lock()

    def _rate_limit(self):
        """Apply rate limiting for Finnhub"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - time_since_last)
            self.last_request_time = time.time()

    def get_real_time_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote from Finnhub"""
        if self.api_key == 'demo':
            return None

        start_time = time.time()
        success = False

        try:
            self._rate_limit()

            params = {
                'symbol': symbol,
                'token': self.api_key
            }

            response = self.session.get(f"{self.base_url}/quote", params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if 'c' not in data or data['c'] == 0:
                return None

            quote_data = {
                'symbol': symbol,
                'price': float(data['c']),  # Current price
                'change': float(data['d']),  # Change
                'change_percent': float(data['dp']),  # Change percent
                'high': float(data['h']),  # High price of the day
                'low': float(data['l']),  # Low price of the day
                'open': float(data['o']),  # Open price of the day
                'previous_close': float(data['pc']),  # Previous close
                'provider': self.name.value,
                'timestamp': datetime.now()
            }

            success = True
            return quote_data

        except Exception as e:
            logger.debug(f"Finnhub quote failed for {symbol}: {e}")
            return None
        finally:
            response_time = time.time() - start_time
            self.health.record_call(success, response_time, str(e) if not success else None)

    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical data from Finnhub"""
        if self.api_key == 'demo':
            return None

        start_time = time.time()
        success = False

        try:
            self._rate_limit()

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            params = {
                'symbol': symbol,
                'resolution': 'D',  # Daily resolution
                'from': int(start_date.timestamp()),
                'to': int(end_date.timestamp()),
                'token': self.api_key
            }

            response = self.session.get(f"{self.base_url}/stock/candle", params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get('s') != 'ok' or not data.get('t'):
                return None

            df = pd.DataFrame({
                'Date': pd.to_datetime(data['t'], unit='s'),
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Close': data['c'],
                'Volume': data['v']
            })

            df['Symbol'] = symbol
            df['Provider'] = self.name.value

            success = True
            return df

        except Exception as e:
            logger.debug(f"Finnhub history failed for {symbol}: {e}")
            return None
        finally:
            response_time = time.time() - start_time
            self.health.record_call(success, response_time, str(e) if not success else None)


class DemoDataProvider:
    """Demo data provider for testing and fallback"""

    def __init__(self):
        self.name = DataProvider.DEMO
        self.health = ProviderHealthMonitor(self.name)
        # Demo provider is always healthy
        self.health.current_status = ProviderStatus.HEALTHY

    def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Generate demo quote data"""
        # Simulate realistic price movement
        base_price = hash(symbol) % 1000 + 100  # Deterministic base price
        price_variation = (time.time() % 10) / 10 * 0.02  # 2% daily variation
        current_price = base_price * (1 + price_variation)

        quote_data = {
            'symbol': symbol,
            'price': round(current_price, 2),
            'change': round(current_price * 0.01, 2),
            'change_percent': 1.0,
            'high': round(current_price * 1.02, 2),
            'low': round(current_price * 0.98, 2),
            'volume': 1000000 + (hash(symbol) % 500000),
            'provider': self.name.value,
            'timestamp': datetime.now()
        }

        self.health.record_call(True, 0.001)  # Always succeeds quickly
        return quote_data

    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Generate demo historical data"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        base_price = hash(symbol) % 1000 + 100

        # Generate realistic OHLCV data
        data = []
        for i, date in enumerate(dates):
            price_trend = base_price * (1 + i * 0.001)  # Slight upward trend
            daily_variation = np.random.normal(0, 0.02)  # 2% daily volatility

            open_price = price_trend * (1 + daily_variation)
            high_price = open_price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.01)))
            close_price = open_price * (1 + np.random.normal(0, 0.01))
            volume = 1000000 + np.random.randint(-200000, 200000)

            data.append({
                'Date': date,
                'Open': round(open_price, 2),
                'High': round(high_price, 2),
                'Low': round(low_price, 2),
                'Close': round(close_price, 2),
                'Volume': volume,
                'Symbol': symbol,
                'Provider': self.name.value
            })

        df = pd.DataFrame(data)
        self.health.record_call(True, 0.001)
        return df


class MultiProviderDataSystem:
    """Multi-provider data system with intelligent fallback"""

    def __init__(self, config: Dict[str, str] = None):
        """
        Initialize multi-provider system

        Args:
            config: Dictionary with API keys for providers
        """
        self.config = config or {}

        # Initialize providers
        self.providers = {
            DataProvider.YAHOO_FINANCE: YahooFinanceProvider(),
            DataProvider.ALPHA_VANTAGE: AlphaVantageProvider(
                self.config.get('alpha_vantage_api_key')
            ),
            DataProvider.FINNHUB: FinnhubProvider(
                self.config.get('finnhub_api_key')
            ),
            DataProvider.DEMO: DemoDataProvider()
        }

        # Provider priority order (can be dynamic based on health)
        self.priority_order = [
            DataProvider.YAHOO_FINANCE,
            DataProvider.ALPHA_VANTAGE,
            DataProvider.FINNHUB,
            DataProvider.DEMO
        ]

        self.cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = 30  # 30 seconds for real-time quotes
        self.lock = threading.Lock()

        logger.info("Multi-provider data system initialized")

    def get_provider_rankings(self) -> List[Tuple[DataProvider, float]]:
        """Get providers ranked by health score"""
        rankings = []
        for provider_type, provider in self.providers.items():
            health_score = provider.health.get_health_score()
            rankings.append((provider_type, health_score))

        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def get_best_available_providers(self) -> List[DataProvider]:
        """Get list of available providers ordered by health"""
        available = []
        for provider_type, provider in self.providers.items():
            if provider.health.is_available():
                available.append(provider_type)

        # Sort by health score
        health_scores = {pt: p.health.get_health_score()
                         for pt, p in self.providers.items() if pt in available}

        return sorted(available, key=lambda x: health_scores[x], reverse=True)

    def get_real_time_quote(self, symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Get real-time quote with provider fallback"""
        cache_key = f"quote_{symbol}"

        # Check cache first
        if use_cache:
            with self.lock:
                if cache_key in self.cache:
                    age = time.time() - self.cache_timestamps[cache_key]
                    if age < self.cache_ttl:
                        return self.cache[cache_key]

        # Try providers in order of health
        available_providers = self.get_best_available_providers()

        for provider_type in available_providers:
            provider = self.providers[provider_type]

            try:
                quote_data = provider.get_real_time_quote(symbol)
                if quote_data:
                    # Cache successful result
                    with self.lock:
                        self.cache[cache_key] = quote_data
                        self.cache_timestamps[cache_key] = time.time()

                    logger.debug(f"Quote for {symbol} obtained from {provider_type.value}")
                    return quote_data

            except Exception as e:
                logger.debug(f"Provider {provider_type.value} failed for {symbol}: {e}")
                continue

        logger.warning(f"All providers failed for quote: {symbol}")
        return None

    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical data with provider fallback"""
        cache_key = f"history_{symbol}_{days}"

        # Check cache (longer TTL for historical data)
        if cache_key in self.cache:
            age = time.time() - self.cache_timestamps[cache_key]
            if age < 300:  # 5 minutes for historical data
                return self.cache[cache_key]

        # Try providers in order of health
        available_providers = self.get_best_available_providers()

        for provider_type in available_providers:
            provider = self.providers[provider_type]

            try:
                hist_data = provider.get_historical_data(symbol, days)
                if hist_data is not None and not hist_data.empty:
                    # Cache successful result
                    with self.lock:
                        self.cache[cache_key] = hist_data
                        self.cache_timestamps[cache_key] = time.time()

                    logger.debug(f"Historical data for {symbol} obtained from {provider_type.value}")
                    return hist_data

            except Exception as e:
                logger.debug(f"Provider {provider_type.value} failed for historical {symbol}: {e}")
                continue

        logger.warning(f"All providers failed for historical data: {symbol}")
        return None

    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols efficiently"""
        results = {}

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(self.get_real_time_quote, symbol): symbol
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

        return results

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        provider_health = {}
        for provider_type, provider in self.providers.items():
            provider_health[provider_type.value] = {
                'status': provider.health.current_status.value,
                'success_rate': provider.health.get_success_rate(),
                'health_score': provider.health.get_health_score(),
                'total_calls': provider.health.total_calls,
                'avg_response_time': provider.health.average_response_time,
                'last_successful_call': provider.health.last_successful_call,
                'available': provider.health.is_available()
            }

        rankings = self.get_provider_rankings()
        available_count = len(self.get_best_available_providers())

        return {
            'provider_health': provider_health,
            'provider_rankings': [(p.value, score) for p, score in rankings],
            'available_providers': available_count,
            'total_providers': len(self.providers),
            'system_status': 'healthy' if available_count >= 2 else 'degraded' if available_count >= 1 else 'failed',
            'cache_entries': len(self.cache)
        }

    def clear_cache(self):
        """Clear all cached data"""
        with self.lock:
            self.cache.clear()
            self.cache_timestamps.clear()
        logger.info("Cache cleared")


def test_multi_provider_system():
    """Test the multi-provider data system"""
    print("Testing Multi-Provider Data System...")

    # Initialize with demo keys (real keys would be in environment)
    config = {
        'alpha_vantage_api_key': 'demo',  # Replace with real key
        'finnhub_api_key': 'demo'  # Replace with real key
    }

    system = MultiProviderDataSystem(config)

    # Test single quote
    quote = system.get_real_time_quote("AAPL")
    if quote:
        print(f"AAPL Quote: ${quote['price']:.2f} from {quote['provider']}")

    # Test multiple quotes
    symbols = ["AAPL", "GOOGL", "MSFT"]
    quotes = system.get_multiple_quotes(symbols)
    print(f"Multi-quote success: {len(quotes)}/{len(symbols)} symbols")

    # Test historical data
    hist_data = system.get_historical_data("AAPL", days=10)
    if hist_data is not None:
        print(f"Historical data: {len(hist_data)} records from {hist_data['Provider'].iloc[0]}")

    # Show system health
    health = system.get_system_health()
    print(f"System status: {health['system_status']}")
    print(f"Available providers: {health['available_providers']}/{health['total_providers']}")

    return system


if __name__ == "__main__":
    system = test_multi_provider_system()