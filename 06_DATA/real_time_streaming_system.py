# 06_DATA/real_time_streaming_system.py
"""
Real-Time Data Streaming System
Continuous market data updates with WebSocket and polling support

Features:
- Real-time quote streaming
- WebSocket connections for live data
- Fallback to HTTP polling when WebSockets unavailable
- Data aggregation and distribution
- Subscription management for symbols
- Real-time analytics and alerts
- Stream health monitoring
- Data quality validation

Location: #06_DATA/real_time_streaming_system.py
"""

import asyncio
import websockets
import json
import threading
import time
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))
sys.path.append(str(current_dir))

# Import our multi-provider system
try:
    from multi_provider_data_system import MultiProviderDataSystem, DataProvider
except ImportError:
    # Fallback for testing
    MultiProviderDataSystem = None
    DataProvider = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Types of data streams"""
    REAL_TIME_QUOTES = "real_time_quotes"
    TICK_DATA = "tick_data"
    ORDER_BOOK = "order_book"
    NEWS = "news"
    ANALYTICS = "analytics"


class StreamStatus(Enum):
    """Stream connection status"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class StreamingQuote:
    """Real-time quote data structure"""
    symbol: str
    price: float
    bid: Optional[float]
    ask: Optional[float]
    volume: int
    change: float
    change_percent: float
    timestamp: datetime
    provider: str
    stream_type: str = "quote"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class StreamSubscription:
    """Stream subscription configuration"""
    symbols: Set[str]
    stream_types: Set[StreamType]
    callback: Callable
    subscription_id: str
    active: bool = True
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class StreamHealthMonitor:
    """Monitor streaming connection health"""

    def __init__(self):
        self.status = StreamStatus.DISCONNECTED
        self.last_data_received = None
        self.messages_received = 0
        self.connection_attempts = 0
        self.reconnection_count = 0
        self.latency_samples = []
        self.error_count = 0
        self.lock = threading.Lock()

    def record_message(self, latency_ms: float = None):
        """Record received message"""
        with self.lock:
            self.messages_received += 1
            self.last_data_received = datetime.now()
            if latency_ms:
                self.latency_samples.append(latency_ms)
                # Keep only last 100 samples
                if len(self.latency_samples) > 100:
                    self.latency_samples.pop(0)

    def record_connection_attempt(self):
        """Record connection attempt"""
        with self.lock:
            self.connection_attempts += 1

    def record_reconnection(self):
        """Record reconnection"""
        with self.lock:
            self.reconnection_count += 1

    def record_error(self):
        """Record error"""
        with self.lock:
            self.error_count += 1

    def get_health_stats(self) -> Dict[str, Any]:
        """Get health statistics"""
        with self.lock:
            avg_latency = np.mean(self.latency_samples) if self.latency_samples else 0
            return {
                'status': self.status.value,
                'messages_received': self.messages_received,
                'last_data_received': self.last_data_received.isoformat() if self.last_data_received else None,
                'connection_attempts': self.connection_attempts,
                'reconnection_count': self.reconnection_count,
                'error_count': self.error_count,
                'average_latency_ms': round(avg_latency, 2),
                'is_healthy': self._is_healthy()
            }

    def _is_healthy(self) -> bool:
        """Check if stream is healthy"""
        if self.status != StreamStatus.CONNECTED:
            return False

        if self.last_data_received:
            time_since_data = (datetime.now() - self.last_data_received).total_seconds()
            return time_since_data < 60  # Healthy if data received within last minute

        return False


class WebSocketStreamer:
    """WebSocket-based real-time data streamer"""

    def __init__(self, url: str):
        self.url = url
        self.websocket = None
        self.health = StreamHealthMonitor()
        self.subscriptions = {}
        self.is_running = False
        self.reconnect_delay = 5
        self.max_reconnect_delay = 60
        self.message_queue = queue.Queue()

    async def connect(self):
        """Connect to WebSocket"""
        try:
            self.health.status = StreamStatus.CONNECTING
            self.health.record_connection_attempt()

            self.websocket = await websockets.connect(self.url)
            self.health.status = StreamStatus.CONNECTED
            logger.info(f"Connected to WebSocket: {self.url}")

        except Exception as e:
            self.health.status = StreamStatus.ERROR
            self.health.record_error()
            logger.error(f"WebSocket connection failed: {e}")
            raise

    async def subscribe_symbol(self, symbol: str):
        """Subscribe to symbol updates"""
        if self.websocket:
            subscribe_message = {
                "action": "subscribe",
                "symbol": symbol,
                "type": "quote"
            }
            await self.websocket.send(json.dumps(subscribe_message))
            logger.debug(f"Subscribed to {symbol}")

    async def listen(self):
        """Listen for incoming messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    self.health.record_message()
                    self.message_queue.put(data)

                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    self.health.record_error()

        except websockets.exceptions.ConnectionClosed:
            self.health.status = StreamStatus.DISCONNECTED
            logger.warning("WebSocket connection closed")
        except Exception as e:
            self.health.status = StreamStatus.ERROR
            self.health.record_error()
            logger.error(f"WebSocket listen error: {e}")

    async def disconnect(self):
        """Disconnect from WebSocket"""
        if self.websocket:
            await self.websocket.close()
            self.health.status = StreamStatus.DISCONNECTED


class PollingStreamer:
    """HTTP polling-based data streamer as WebSocket fallback"""

    def __init__(self, data_provider: MultiProviderDataSystem):
        self.data_provider = data_provider
        self.health = StreamHealthMonitor()
        self.subscribed_symbols = set()
        self.polling_interval = 5  # 5 seconds
        self.is_running = False
        self.polling_thread = None
        self.message_queue = queue.Queue()

    def start_polling(self):
        """Start polling for data updates"""
        if self.is_running:
            return

        self.is_running = True
        self.health.status = StreamStatus.CONNECTED
        self.polling_thread = threading.Thread(target=self._polling_loop, daemon=True)
        self.polling_thread.start()
        logger.info("Started polling data streamer")

    def stop_polling(self):
        """Stop polling"""
        self.is_running = False
        self.health.status = StreamStatus.DISCONNECTED
        if self.polling_thread:
            self.polling_thread.join()
        logger.info("Stopped polling data streamer")

    def subscribe_symbol(self, symbol: str):
        """Subscribe to symbol updates"""
        self.subscribed_symbols.add(symbol)
        logger.debug(f"Added {symbol} to polling subscription")

    def unsubscribe_symbol(self, symbol: str):
        """Unsubscribe from symbol updates"""
        self.subscribed_symbols.discard(symbol)
        logger.debug(f"Removed {symbol} from polling subscription")

    def _polling_loop(self):
        """Main polling loop"""
        while self.is_running:
            try:
                if self.subscribed_symbols:
                    start_time = time.time()

                    # Get quotes for all subscribed symbols
                    quotes = self.data_provider.get_multiple_quotes(list(self.subscribed_symbols))

                    # Queue the updates
                    for symbol, quote_data in quotes.items():
                        streaming_quote = StreamingQuote(
                            symbol=symbol,
                            price=quote_data['price'],
                            bid=quote_data.get('bid'),
                            ask=quote_data.get('ask'),
                            volume=quote_data.get('volume', 0),
                            change=quote_data.get('change', 0),
                            change_percent=quote_data.get('change_percent', 0),
                            timestamp=quote_data['timestamp'],
                            provider=quote_data['provider']
                        )

                        self.message_queue.put(streaming_quote.to_dict())

                    latency = (time.time() - start_time) * 1000
                    self.health.record_message(latency)

                time.sleep(self.polling_interval)

            except Exception as e:
                logger.error(f"Polling error: {e}")
                self.health.record_error()
                time.sleep(self.polling_interval)


class RealTimeStreamingSystem:
    """Main real-time streaming system"""

    def __init__(self, data_provider: MultiProviderDataSystem = None, db_path: str = None):
        """
        Initialize streaming system

        Args:
            data_provider: Multi-provider data system for fallback
            db_path: Database path for storing streaming data
        """
        self.data_provider = data_provider or MultiProviderDataSystem()
        self.db_path = db_path or "marketpulse_production.db"

        # Streaming components
        self.websocket_streamer = None
        self.polling_streamer = PollingStreamer(self.data_provider)
        self.active_streamer = None

        # Subscription management
        self.subscriptions = {}
        self.subscription_counter = 0
        self.lock = threading.Lock()

        # Data processing
        self.data_processor = None
        self.is_running = False

        # Initialize database
        self._init_streaming_database()

        logger.info("Real-time streaming system initialized")

    def _init_streaming_database(self):
        """Initialize database for streaming data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Real-time quotes stream
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS streaming_quotes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        price REAL NOT NULL,
                        bid REAL,
                        ask REAL,
                        volume INTEGER,
                        change_amount REAL,
                        change_percent REAL,
                        provider TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        received_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Stream analytics
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS stream_analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        calculation_window TEXT,
                        timestamp DATETIME NOT NULL
                    )
                """)

                # Create indexes
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_streaming_symbol_time ON streaming_quotes(symbol, timestamp)")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_analytics_symbol_type ON stream_analytics(symbol, metric_type)")

                conn.commit()
                logger.info("Streaming database initialized")

        except Exception as e:
            logger.error(f"Failed to initialize streaming database: {e}")

    def start_streaming(self, websocket_url: str = None, use_websockets: bool = True):
        """Start real-time streaming"""
        if self.is_running:
            logger.warning("Streaming already running")
            return

        self.is_running = True

        # Try WebSocket connection first if URL provided
        if use_websockets and websocket_url:
            try:
                self.websocket_streamer = WebSocketStreamer(websocket_url)
                # Note: WebSocket connection would be async in real implementation
                self.active_streamer = self.websocket_streamer
                logger.info("Using WebSocket streaming")
            except Exception as e:
                logger.warning(f"WebSocket failed, falling back to polling: {e}")
                self.active_streamer = self.polling_streamer
        else:
            self.active_streamer = self.polling_streamer

        # Start the active streamer
        if self.active_streamer == self.polling_streamer:
            self.polling_streamer.start_polling()

        # Start data processor
        self.data_processor = threading.Thread(target=self._process_streaming_data, daemon=True)
        self.data_processor.start()

        logger.info("Real-time streaming started")

    def stop_streaming(self):
        """Stop real-time streaming"""
        self.is_running = False

        if self.polling_streamer.is_running:
            self.polling_streamer.stop_polling()

        if self.data_processor:
            self.data_processor.join(timeout=5)

        logger.info("Real-time streaming stopped")

    def subscribe(self, symbols: List[str], stream_types: List[StreamType],
                  callback: Callable[[Dict[str, Any]], None]) -> str:
        """
        Subscribe to real-time data streams

        Args:
            symbols: List of symbols to subscribe to
            stream_types: Types of streams to subscribe to
            callback: Function to call when data received

        Returns:
            Subscription ID
        """
        with self.lock:
            subscription_id = f"sub_{self.subscription_counter}"
            self.subscription_counter += 1

            subscription = StreamSubscription(
                symbols=set(symbols),
                stream_types=set(stream_types),
                callback=callback,
                subscription_id=subscription_id
            )

            self.subscriptions[subscription_id] = subscription

            # Add symbols to active streamer
            for symbol in symbols:
                if self.active_streamer == self.polling_streamer:
                    self.polling_streamer.subscribe_symbol(symbol)
                # WebSocket subscription would be handled differently

        logger.info(f"Created subscription {subscription_id} for {len(symbols)} symbols")
        return subscription_id

    def unsubscribe(self, subscription_id: str):
        """Unsubscribe from data streams"""
        with self.lock:
            if subscription_id in self.subscriptions:
                subscription = self.subscriptions[subscription_id]
                subscription.active = False

                # Remove symbols if no other subscriptions
                symbols_to_remove = set()
                for symbol in subscription.symbols:
                    still_subscribed = any(
                        sub.active and symbol in sub.symbols
                        for sub_id, sub in self.subscriptions.items()
                        if sub_id != subscription_id
                    )
                    if not still_subscribed:
                        symbols_to_remove.add(symbol)

                # Remove from active streamer
                for symbol in symbols_to_remove:
                    if self.active_streamer == self.polling_streamer:
                        self.polling_streamer.unsubscribe_symbol(symbol)

                del self.subscriptions[subscription_id]
                logger.info(f"Removed subscription {subscription_id}")

    def _process_streaming_data(self):
        """Process incoming streaming data"""
        while self.is_running:
            try:
                # Get message from active streamer queue
                message_queue = (self.polling_streamer.message_queue if
                                 self.active_streamer == self.polling_streamer else None)

                if message_queue:
                    try:
                        message = message_queue.get(timeout=1)
                        self._handle_streaming_message(message)
                    except queue.Empty:
                        continue
                else:
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error processing streaming data: {e}")
                time.sleep(1)

    def _handle_streaming_message(self, message: Dict[str, Any]):
        """Handle incoming streaming message"""
        try:
            # Store in database
            self._store_streaming_data(message)

            # Calculate real-time analytics
            self._calculate_streaming_analytics(message)

            # Notify subscribers
            self._notify_subscribers(message)

        except Exception as e:
            logger.error(f"Error handling streaming message: {e}")

    def _store_streaming_data(self, message: Dict[str, Any]):
        """Store streaming data in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO streaming_quotes 
                    (symbol, price, bid, ask, volume, change_amount, change_percent, provider, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message['symbol'],
                    message['price'],
                    message.get('bid'),
                    message.get('ask'),
                    message.get('volume'),
                    message.get('change'),
                    message.get('change_percent'),
                    message['provider'],
                    message['timestamp']
                ))

        except Exception as e:
            logger.error(f"Failed to store streaming data: {e}")

    def _calculate_streaming_analytics(self, message: Dict[str, Any]):
        """Calculate real-time analytics"""
        symbol = message['symbol']
        price = message['price']
        timestamp = message['timestamp']

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Calculate 1-minute moving average
                one_minute_ago = (datetime.fromisoformat(timestamp) - timedelta(minutes=1)).isoformat()

                cursor = conn.execute("""
                    SELECT AVG(price) FROM streaming_quotes 
                    WHERE symbol = ? AND timestamp >= ?
                """, (symbol, one_minute_ago))

                avg_price = cursor.fetchone()[0]
                if avg_price:
                    conn.execute("""
                        INSERT INTO stream_analytics 
                        (symbol, metric_type, metric_value, calculation_window, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    """, (symbol, 'moving_average_1min', avg_price, '1min', timestamp))

                # Calculate price change from previous
                cursor = conn.execute("""
                    SELECT price FROM streaming_quotes 
                    WHERE symbol = ? AND timestamp < ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (symbol, timestamp))

                prev_price = cursor.fetchone()
                if prev_price:
                    price_change = ((price - prev_price[0]) / prev_price[0]) * 100
                    conn.execute("""
                        INSERT INTO stream_analytics 
                        (symbol, metric_type, metric_value, calculation_window, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    """, (symbol, 'price_change_pct', price_change, 'tick', timestamp))

        except Exception as e:
            logger.error(f"Failed to calculate analytics: {e}")

    def _notify_subscribers(self, message: Dict[str, Any]):
        """Notify all relevant subscribers"""
        symbol = message['symbol']

        for subscription_id, subscription in self.subscriptions.items():
            if (subscription.active and
                    symbol in subscription.symbols and
                    StreamType.REAL_TIME_QUOTES in subscription.stream_types):

                try:
                    subscription.callback(message)
                except Exception as e:
                    logger.error(f"Subscriber callback error: {e}")

    def get_streaming_health(self) -> Dict[str, Any]:
        """Get streaming system health status"""
        active_streamer_health = None
        if self.active_streamer:
            if self.active_streamer == self.polling_streamer:
                active_streamer_health = self.polling_streamer.health.get_health_stats()
            elif self.websocket_streamer:
                active_streamer_health = self.websocket_streamer.health.get_health_stats()

        return {
            'is_running': self.is_running,
            'active_streamer': 'polling' if self.active_streamer == self.polling_streamer else 'websocket',
            'active_subscriptions': len([s for s in self.subscriptions.values() if s.active]),
            'total_symbols': len(set().union(*[s.symbols for s in self.subscriptions.values() if s.active])),
            'streamer_health': active_streamer_health
        }

    def get_recent_data(self, symbol: str, minutes: int = 5) -> pd.DataFrame:
        """Get recent streaming data for analysis"""
        try:
            cutoff_time = (datetime.now() - timedelta(minutes=minutes)).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("""
                    SELECT * FROM streaming_quotes 
                    WHERE symbol = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """, conn, params=(symbol, cutoff_time))

                return df

        except Exception as e:
            logger.error(f"Failed to get recent data: {e}")
            return pd.DataFrame()


def test_streaming_system():
    """Test the real-time streaming system"""
    print("Testing Real-Time Streaming System...")

    # Initialize with multi-provider system
    if MultiProviderDataSystem:
        data_provider = MultiProviderDataSystem()
    else:
        data_provider = None
        print("Note: Multi-provider system not available for full test")

    streaming_system = RealTimeStreamingSystem(data_provider)

    # Define callback for received data
    def data_callback(data):
        print(f"Received: {data['symbol']} @ ${data['price']:.2f} from {data['provider']}")

    # Subscribe to some symbols
    subscription_id = streaming_system.subscribe(
        symbols=["AAPL", "GOOGL", "MSFT"],
        stream_types=[StreamType.REAL_TIME_QUOTES],
        callback=data_callback
    )

    # Start streaming
    streaming_system.start_streaming(use_websockets=False)  # Use polling for demo

    print("Streaming started - will run for 30 seconds...")
    time.sleep(30)

    # Get health status
    health = streaming_system.get_streaming_health()
    print(f"System health: {health}")

    # Get recent data
    recent_data = streaming_system.get_recent_data("AAPL", minutes=5)
    print(f"Recent AAPL data points: {len(recent_data)}")

    # Stop streaming
    streaming_system.unsubscribe(subscription_id)
    streaming_system.stop_streaming()

    print("Streaming test completed")
    return streaming_system


if __name__ == "__main__":
    system = test_streaming_system()