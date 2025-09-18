# src/data/streaming/websocket_service.py
"""
WebSocket Streaming Service for Phase 1 Day 8
Real-time market data streaming with AI enhancement for MarketPulse

Features:
- WebSocket server for real-time data distribution
- AI-powered market alerts and notifications
- Multi-client support with different subscription levels
- Real-time AI analysis and signal generation
- Performance monitoring and automatic scaling
"""

import asyncio
import websockets
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, asdict
import uuid
from enum import Enum
import weakref
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubscriptionLevel(Enum):
    """Client subscription levels"""
    BASIC = "basic"  # Price updates only
    ADVANCED = "advanced"  # Price + volume + technical indicators
    PREMIUM = "premium"  # Everything + AI analysis + alerts


@dataclass
class ClientSession:
    """WebSocket client session information"""
    id: str
    websocket: Any
    symbols: Set[str]
    subscription_level: SubscriptionLevel
    connected_at: datetime
    last_activity: datetime
    message_count: int = 0


@dataclass
class MarketAlert:
    """AI-generated market alert"""
    alert_id: str
    symbol: str
    alert_type: str  # 'breakout', 'volume_spike', 'price_target', 'risk_warning'
    message: str
    confidence_score: float
    timestamp: datetime
    ai_provider: str
    priority: int  # 1=low, 2=medium, 3=high, 4=critical


class WebSocketStreamingService:
    """
    Enterprise WebSocket streaming service for real-time market data

    Features:
    - Multi-client WebSocket connections
    - Subscription-based data filtering
    - AI-powered real-time analysis and alerts
    - Performance monitoring and auto-scaling
    - Graceful error handling and recovery
    """

    def __init__(self, ai_framework=None, port: int = 8765):
        """Initialize WebSocket streaming service"""
        self.ai_framework = ai_framework
        self.port = port
        self.clients: Dict[str, ClientSession] = {}
        self.server = None
        self.is_running = False

        # Data cache for latest market data
        self.market_data_cache = {}
        self.ai_analysis_cache = {}
        self.alert_history = []

        # Performance metrics
        self.metrics = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'ai_alerts_generated': 0,
            'average_response_time_ms': 0,
            'uptime_seconds': 0
        }

        self.start_time = time.time()

        # AI alert generation settings
        self.alert_settings = {
            'price_change_threshold': 2.0,  # Alert on >2% price moves
            'volume_spike_threshold': 2.0,  # Alert on 2x average volume
            'ai_confidence_threshold': 0.7,  # Only high-confidence AI alerts
            'max_alerts_per_symbol_hour': 5  # Rate limiting
        }

    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"🚀 Starting WebSocket server on port {self.port}")

        try:
            self.server = await websockets.serve(
                self._handle_client,
                "localhost",
                self.port,
                ping_interval=30,  # Send ping every 30 seconds
                ping_timeout=10,  # Wait 10 seconds for pong
                max_size=1000000,  # 1MB message size limit
                compression=None  # Disable compression for speed
            )

            self.is_running = True
            logger.info(f"✅ WebSocket server started on ws://localhost:{self.port}")

            # Start background tasks
            await asyncio.gather(
                self._performance_monitor(),
                self._ai_analysis_loop(),
                self._cleanup_inactive_clients()
            )

        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket server: {e}")
            raise

    async def _handle_client(self, websocket, path):
        """Handle new WebSocket client connection"""
        client_id = str(uuid.uuid4())[:8]
        client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"

        logger.info(f"🔌 New client connected: {client_id} from {client_ip}")

        try:
            # Create client session
            session = ClientSession(
                id=client_id,
                websocket=websocket,
                symbols=set(),
                subscription_level=SubscriptionLevel.BASIC,
                connected_at=datetime.now(timezone.utc),
                last_activity=datetime.now(timezone.utc)
            )

            self.clients[client_id] = session
            self.metrics['total_connections'] += 1
            self.metrics['active_connections'] = len(self.clients)

            # Send welcome message
            await self._send_to_client(client_id, {
                'type': 'connection_established',
                'client_id': client_id,
                'server_time': datetime.now(timezone.utc).isoformat(),
                'available_symbols': list(self.market_data_cache.keys()),
                'subscription_levels': [level.value for level in SubscriptionLevel]
            })

            # Handle client messages
            async for message in websocket:
                await self._handle_client_message(client_id, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Client {client_id} disconnected normally")
        except Exception as e:
            logger.error(f"❌ Client {client_id} error: {e}")
        finally:
            # Cleanup client session
            if client_id in self.clients:
                del self.clients[client_id]
                self.metrics['active_connections'] = len(self.clients)
                logger.info(f"🧹 Cleaned up client session: {client_id}")

    async def _handle_client_message(self, client_id: str, message: str):
        """Handle incoming message from client"""
        try:
            data = json.loads(message)
            session = self.clients.get(client_id)

            if not session:
                return

            # Update last activity
            session.last_activity = datetime.now(timezone.utc)
            session.message_count += 1

            message_type = data.get('type')

            if message_type == 'subscribe':
                await self._handle_subscribe(client_id, data)
            elif message_type == 'unsubscribe':
                await self._handle_unsubscribe(client_id, data)
            elif message_type == 'set_subscription_level':
                await self._handle_subscription_level(client_id, data)
            elif message_type == 'request_historical':
                await self._handle_historical_request(client_id, data)
            elif message_type == 'ai_analysis_request':
                await self._handle_ai_analysis_request(client_id, data)
            else:
                await self._send_error(client_id, f"Unknown message type: {message_type}")

        except json.JSONDecodeError as e:
            await self._send_error(client_id, f"Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
            await self._send_error(client_id, f"Message processing error: {e}")

    async def _handle_subscribe(self, client_id: str, data: Dict):
        """Handle symbol subscription request"""
        symbols = data.get('symbols', [])
        session = self.clients.get(client_id)

        if not session:
            return

        # Validate symbols
        valid_symbols = []
        for symbol in symbols:
            if isinstance(symbol, str) and len(symbol) > 0:
                valid_symbols.append(symbol.upper())

        # Update subscription
        session.symbols.update(valid_symbols)

        await self._send_to_client(client_id, {
            'type': 'subscription_confirmed',
            'symbols': valid_symbols,
            'total_subscribed': len(session.symbols)
        })

        # Send latest data for newly subscribed symbols
        for symbol in valid_symbols:
            if symbol in self.market_data_cache:
                data_point = self.market_data_cache[symbol]
                await self._send_market_data_to_client(client_id, data_point)

        logger.info(f"📊 Client {client_id} subscribed to {len(valid_symbols)} symbols")

    async def _handle_unsubscribe(self, client_id: str, data: Dict):
        """Handle symbol unsubscription request"""
        symbols = data.get('symbols', [])
        session = self.clients.get(client_id)

        if not session:
            return

        # Remove symbols from subscription
        for symbol in symbols:
            session.symbols.discard(symbol.upper())

        await self._send_to_client(client_id, {
            'type': 'unsubscription_confirmed',
            'symbols': symbols,
            'remaining_subscribed': len(session.symbols)
        })

        logger.info(f"📊 Client {client_id} unsubscribed from {len(symbols)} symbols")

    async def _handle_subscription_level(self, client_id: str, data: Dict):
        """Handle subscription level change"""
        level_str = data.get('level', 'basic')
        session = self.clients.get(client_id)

        if not session:
            return

        try:
            new_level = SubscriptionLevel(level_str)
            session.subscription_level = new_level

            await self._send_to_client(client_id, {
                'type': 'subscription_level_updated',
                'level': new_level.value,
                'features': self._get_subscription_features(new_level)
            })

            logger.info(f"⬆️ Client {client_id} upgraded to {new_level.value}")

        except ValueError:
            await self._send_error(client_id, f"Invalid subscription level: {level_str}")

    async def _handle_historical_request(self, client_id: str, data: Dict):
        """Handle historical data request"""
        symbol = data.get('symbol', '').upper()
        period = data.get('period', '1d')

        if not symbol:
            await self._send_error(client_id, "Symbol is required for historical data")
            return

        try:
            # This would integrate with the RealTimeDataCollector
            # For now, send a placeholder response
            await self._send_to_client(client_id, {
                'type': 'historical_data',
                'symbol': symbol,
                'period': period,
                'status': 'requested',
                'message': 'Historical data request received and processing'
            })

        except Exception as e:
            await self._send_error(client_id, f"Historical data error: {e}")

    async def _handle_ai_analysis_request(self, client_id: str, data: Dict):
        """Handle AI analysis request"""
        session = self.clients.get(client_id)
        if not session or session.subscription_level == SubscriptionLevel.BASIC:
            await self._send_error(client_id, "AI analysis requires Advanced or Premium subscription")
            return

        symbol = data.get('symbol', '').upper()
        analysis_type = data.get('analysis_type', 'comprehensive')

        if not symbol:
            await self._send_error(client_id, "Symbol is required for AI analysis")
            return

        if not self.ai_framework:
            await self._send_error(client_id, "AI analysis not available")
            return

        try:
            # Generate AI analysis
            analysis = await self._generate_ai_analysis(symbol, analysis_type)

            await self._send_to_client(client_id, {
                'type': 'ai_analysis',
                'symbol': symbol,
                'analysis_type': analysis_type,
                'analysis': analysis,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

            logger.info(f"🤖 AI analysis delivered to client {client_id} for {symbol}")

        except Exception as e:
            await self._send_error(client_id, f"AI analysis error: {e}")

    def _get_subscription_features(self, level: SubscriptionLevel) -> List[str]:
        """Get features available for subscription level"""
        features = {
            SubscriptionLevel.BASIC: [
                "Real-time price updates",
                "Basic market data"
            ],
            SubscriptionLevel.ADVANCED: [
                "Real-time price updates",
                "Volume and technical indicators",
                "Market alerts",
                "Historical data access"
            ],
            SubscriptionLevel.PREMIUM: [
                "All Advanced features",
                "AI-powered analysis",
                "Custom alerts",
                "Priority support",
                "Advanced risk metrics"
            ]
        }
        return features.get(level, [])

    async def broadcast_market_data(self, data_point):
        """Broadcast market data to all subscribed clients"""
        if not self.clients:
            return

        # Update cache
        self.market_data_cache[data_point.symbol] = data_point

        # Find clients subscribed to this symbol
        target_clients = []
        for client_id, session in self.clients.items():
            if data_point.symbol in session.symbols:
                target_clients.append(client_id)

        if not target_clients:
            return

        # Send to all target clients concurrently
        tasks = []
        for client_id in target_clients:
            task = asyncio.create_task(
                self._send_market_data_to_client(client_id, data_point)
            )
            tasks.append(task)

        # Wait for all sends to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self.metrics['messages_sent'] += len(tasks)

    async def _send_market_data_to_client(self, client_id: str, data_point):
        """Send market data to specific client based on subscription level"""
        session = self.clients.get(client_id)
        if not session:
            return

        # Prepare data based on subscription level
        if session.subscription_level == SubscriptionLevel.BASIC:
            message = {
                'type': 'market_data',
                'symbol': data_point.symbol,
                'price': data_point.price,
                'timestamp': data_point.timestamp.isoformat()
            }
        elif session.subscription_level == SubscriptionLevel.ADVANCED:
            message = {
                'type': 'market_data',
                'symbol': data_point.symbol,
                'price': data_point.price,
                'volume': data_point.volume,
                'change_percent': data_point.change_percent,
                'bid': data_point.bid,
                'ask': data_point.ask,
                'quality_score': data_point.quality_score,
                'timestamp': data_point.timestamp.isoformat()
            }
        else:  # PREMIUM
            # Include AI analysis if available
            ai_analysis = self.ai_analysis_cache.get(data_point.symbol, {})
            message = {
                'type': 'market_data',
                'symbol': data_point.symbol,
                'price': data_point.price,
                'volume': data_point.volume,
                'change_percent': data_point.change_percent,
                'bid': data_point.bid,
                'ask': data_point.ask,
                'quality_score': data_point.quality_score,
                'ai_analysis': ai_analysis,
                'timestamp': data_point.timestamp.isoformat()
            }

        await self._send_to_client(client_id, message)

    async def broadcast_ai_alert(self, alert: MarketAlert):
        """Broadcast AI-generated alert to subscribed clients"""
        if not self.clients:
            return

        # Store alert in history
        self.alert_history.append(alert)
        if len(self.alert_history) > 1000:  # Keep last 1000 alerts
            self.alert_history.pop(0)

        # Find clients with premium subscription and symbol subscription
        target_clients = []
        for client_id, session in self.clients.items():
            if (session.subscription_level != SubscriptionLevel.BASIC and
                    alert.symbol in session.symbols):
                target_clients.append(client_id)

        if not target_clients:
            return

        # Prepare alert message
        alert_message = {
            'type': 'ai_alert',
            'alert_id': alert.alert_id,
            'symbol': alert.symbol,
            'alert_type': alert.alert_type,
            'message': alert.message,
            'confidence_score': alert.confidence_score,
            'priority': alert.priority,
            'ai_provider': alert.ai_provider,
            'timestamp': alert.timestamp.isoformat()
        }

        # Send to all target clients
        tasks = []
        for client_id in target_clients:
            task = asyncio.create_task(
                self._send_to_client(client_id, alert_message)
            )
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self.metrics['ai_alerts_generated'] += 1

        logger.info(f"🚨 AI Alert broadcasted: {alert.alert_type} for {alert.symbol} "
                    f"to {len(target_clients)} clients")

    async def _send_to_client(self, client_id: str, message: Dict):
        """Send message to specific client with error handling"""
        session = self.clients.get(client_id)
        if not session:
            return

        try:
            message_json = json.dumps(message)
            await session.websocket.send(message_json)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Client {client_id} connection closed during send")
            # Client will be cleaned up by connection handler
        except Exception as e:
            logger.error(f"❌ Error sending to client {client_id}: {e}")
            # Remove problematic client
            if client_id in self.clients:
                del self.clients[client_id]
                self.metrics['active_connections'] = len(self.clients)

    async def _send_error(self, client_id: str, error_message: str):
        """Send error message to client"""
        await self._send_to_client(client_id, {
            'type': 'error',
            'message': error_message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    async def _performance_monitor(self):
        """Monitor and log performance metrics"""
        while self.is_running:
            try:
                # Update uptime
                self.metrics['uptime_seconds'] = time.time() - self.start_time

                # Log metrics every 60 seconds
                logger.info(f"📊 WebSocket Metrics: "
                            f"Active: {self.metrics['active_connections']}, "
                            f"Total: {self.metrics['total_connections']}, "
                            f"Messages: {self.metrics['messages_sent']}, "
                            f"Uptime: {self.metrics['uptime_seconds'] / 60:.1f}min")

                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(5)

    async def _ai_analysis_loop(self):
        """Background loop for generating AI analysis"""
        while self.is_running:
            try:
                if not self.ai_framework or not self.market_data_cache:
                    await asyncio.sleep(10)
                    continue

                # Generate AI analysis for active symbols
                active_symbols = set()
                for session in self.clients.values():
                    if session.subscription_level == SubscriptionLevel.PREMIUM:
                        active_symbols.update(session.symbols)

                if not active_symbols:
                    await asyncio.sleep(10)
                    continue

                # Process a few symbols at a time
                symbols_to_process = list(active_symbols)[:5]  # Limit to 5 per cycle

                for symbol in symbols_to_process:
                    if symbol in self.market_data_cache:
                        await self._update_ai_analysis_cache(symbol)

                await asyncio.sleep(30)  # Run every 30 seconds

            except Exception as e:
                logger.error(f"AI analysis loop error: {e}")
                await asyncio.sleep(10)

    async def _cleanup_inactive_clients(self):
        """Clean up inactive client connections"""
        while self.is_running:
            try:
                current_time = datetime.now(timezone.utc)
                inactive_clients = []

                for client_id, session in self.clients.items():
                    # Consider client inactive if no activity for 5 minutes
                    inactive_duration = (current_time - session.last_activity).total_seconds()
                    if inactive_duration > 300:  # 5 minutes
                        inactive_clients.append(client_id)

                # Remove inactive clients
                for client_id in inactive_clients:
                    session = self.clients.get(client_id)
                    if session:
                        try:
                            await session.websocket.close()
                        except:
                            pass
                        del self.clients[client_id]
                        logger.info(f"🧹 Removed inactive client: {client_id}")

                if inactive_clients:
                    self.metrics['active_connections'] = len(self.clients)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Client cleanup error: {e}")
                await asyncio.sleep(10)

    async def _generate_ai_analysis(self, symbol: str, analysis_type: str) -> Dict:
        """Generate AI analysis for a symbol"""
        if not self.ai_framework:
            return {"error": "AI framework not available"}

        market_data = self.market_data_cache.get(symbol)
        if not market_data:
            return {"error": "No market data available"}

        try:
            analysis_prompt = f"""
            Provide {analysis_type} analysis for {symbol}:

            Current Data:
            - Price: ₹{market_data.price}
            - Volume: {market_data.volume:,}
            - Change: {market_data.change_percent:+.2f}%
            - Quality Score: {market_data.quality_score:.2f}

            Provide analysis in JSON format:
            {{
                "signal": "BUY/SELL/HOLD",
                "confidence": 0.75,
                "target_price": 1250.00,
                "stop_loss": 1150.00,
                "reasoning": ["key point 1", "key point 2"],
                "risk_level": "LOW/MEDIUM/HIGH",
                "time_horizon": "SHORT/MEDIUM/LONG"
            }}
            """

            ai_response = await asyncio.wait_for(
                self.ai_framework.get_completion(analysis_prompt),
                timeout=10.0
            )

            return json.loads(ai_response)

        except asyncio.TimeoutError:
            return {"error": "AI analysis timeout"}
        except json.JSONDecodeError:
            return {"error": "Invalid AI response format"}
        except Exception as e:
            return {"error": f"AI analysis failed: {str(e)}"}

    async def _update_ai_analysis_cache(self, symbol: str):
        """Update AI analysis cache for a symbol"""
        try:
            analysis = await self._generate_ai_analysis(symbol, "technical")
            if "error" not in analysis:
                self.ai_analysis_cache[symbol] = {
                    **analysis,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }

                # Check if we should generate an alert
                await self._check_for_alert_conditions(symbol, analysis)

        except Exception as e:
            logger.error(f"Error updating AI analysis for {symbol}: {e}")

    async def _check_for_alert_conditions(self, symbol: str, analysis: Dict):
        """Check if conditions warrant generating an alert"""
        try:
            market_data = self.market_data_cache.get(symbol)
            if not market_data:
                return

            # High confidence signals
            if analysis.get('confidence', 0) >= self.alert_settings['ai_confidence_threshold']:
                signal = analysis.get('signal', 'HOLD')
                if signal in ['BUY', 'SELL']:
                    alert = MarketAlert(
                        alert_id=str(uuid.uuid4())[:8],
                        symbol=symbol,
                        alert_type='ai_signal',
                        message=f"AI {signal} signal for {symbol} with {analysis['confidence']:.0%} confidence",
                        confidence_score=analysis['confidence'],
                        timestamp=datetime.now(timezone.utc),
                        ai_provider=getattr(self.ai_framework, 'current_provider', 'AI'),
                        priority=3 if analysis['confidence'] >= 0.9 else 2
                    )

                    await self.broadcast_ai_alert(alert)

            # Price movement alerts
            if abs(market_data.change_percent) >= self.alert_settings['price_change_threshold']:
                direction = "surge" if market_data.change_percent > 0 else "drop"
                alert = MarketAlert(
                    alert_id=str(uuid.uuid4())[:8],
                    symbol=symbol,
                    alert_type='price_movement',
                    message=f"{symbol} {direction}: {market_data.change_percent:+.2f}% to ₹{market_data.price}",
                    confidence_score=1.0,  # Price movements are factual
                    timestamp=datetime.now(timezone.utc),
                    ai_provider='system',
                    priority=3 if abs(market_data.change_percent) >= 5 else 2
                )

                await self.broadcast_ai_alert(alert)

        except Exception as e:
            logger.error(f"Error checking alert conditions for {symbol}: {e}")

    async def stop_server(self):
        """Stop the WebSocket server"""
        logger.info("🛑 Stopping WebSocket server...")
        self.is_running = False

        # Close all client connections
        for client_id, session in self.clients.items():
            try:
                await session.websocket.close()
            except:
                pass

        # Stop server
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        logger.info("✅ WebSocket server stopped")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.metrics,
            'active_symbols': len(self.market_data_cache),
            'cached_analyses': len(self.ai_analysis_cache),
            'alert_history_count': len(self.alert_history)
        }

    def get_client_info(self) -> List[Dict]:
        """Get information about connected clients"""
        return [
            {
                'id': session.id,
                'symbols': list(session.symbols),
                'subscription_level': session.subscription_level.value,
                'connected_at': session.connected_at.isoformat(),
                'message_count': session.message_count
            }
            for session in self.clients.values()
        ]


# Example integration with RealTimeDataCollector
class StreamingDataIntegrator:
    """Integration between RealTimeDataCollector and WebSocketStreamingService"""

    def __init__(self, collector, streaming_service):
        self.collector = collector
        self.streaming_service = streaming_service

    async def start_integrated_streaming(self, symbols: List[str]):
        """Start integrated real-time streaming"""
        # Add WebSocket service as subscriber to data collector
        self.collector.add_subscriber(
            self.streaming_service.broadcast_market_data
        )

        # Start both services
        await asyncio.gather(
            self.collector.start_streaming(symbols, update_interval=2.0),
            self.streaming_service.start_server()
        )


# Example usage
async def example_usage():
    """Example usage of WebSocket streaming service"""

    print("🚀 Starting MarketPulse WebSocket Streaming Service")
    print("📡 Server will be available at ws://localhost:8765")

    # Initialize service
    streaming_service = WebSocketStreamingService(port=8765)

    try:
        await streaming_service.start_server()
    except KeyboardInterrupt:
        print("\n⛔ Stopping server...")
        await streaming_service.stop_server()


if __name__ == "__main__":
    # Test the WebSocket streaming service
    asyncio.run(example_usage())