"""
MarketPulse Telegram Alerts System - Fixed Version
Real-time trading alerts and notifications via Telegram
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Try multiple possible locations
env_locations = [
    Path.cwd() / '.env',  # Current directory
    Path.cwd().parent / '.env',  # Parent directory
    Path(__file__).parent.parent.parent / '.env',  # Project root
]

for env_path in env_locations:
    if env_path.exists():
        load_dotenv(env_path)
        break

# Telegram bot library
try:
    from telegram import Bot
    from telegram.ext import Application, CommandHandler, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("⚠️ Telegram library not installed. Run: pip install python-telegram-bot")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertType(Enum):
    """Types of alerts"""
    TRADE_SIGNAL = "📊 Trade Signal"
    TRADE_EXECUTED = "✅ Trade Executed"
    STOP_LOSS_HIT = "🛑 Stop Loss Hit"
    TARGET_REACHED = "🎯 Target Reached"
    RISK_WARNING = "⚠️ Risk Warning"
    DAILY_SUMMARY = "📈 Daily Summary"
    SYSTEM_STATUS = "🔧 System Status"
    ERROR = "❌ Error"
    INFO = "ℹ️ Info"

@dataclass
class Alert:
    """Alert message structure"""
    alert_type: AlertType
    title: str
    message: str
    data: Optional[Dict] = None
    timestamp: datetime = None
    priority: int = 1  # 1=Low, 2=Medium, 3=High

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def format_message(self) -> str:
        """Format alert for Telegram"""
        formatted = f"{self.alert_type.value}\n"
        formatted += f"*{self.title}*\n"
        formatted += f"⏰ {self.timestamp.strftime('%H:%M:%S')}\n\n"
        formatted += f"{self.message}\n"

        if self.data:
            formatted += "\n📊 *Details:*\n"
            for key, value in self.data.items():
                formatted += f"• {key}: {value}\n"

        return formatted

class TelegramAlertSystem:
    """
    Telegram Alert System for MarketPulse
    Sends real-time trading alerts to Telegram
    """

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initialize Telegram alert system

        Args:
            bot_token: Telegram bot token (or from env)
            chat_id: Telegram chat ID (or from env)
        """
        # Load from environment if not provided
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')

        # Debug logging
        logger.info(f"Telegram Bot Token: {'Set' if self.bot_token else 'Not set'}")
        logger.info(f"Telegram Chat ID: {'Set' if self.chat_id else 'Not set'}")

        # Validate and initialize
        self.enabled = False
        self.bot = None

        if not TELEGRAM_AVAILABLE:
            logger.warning("Telegram library not installed")
        elif not self.bot_token or not self.chat_id:
            logger.warning(f"Telegram credentials missing - Token: {bool(self.bot_token)}, ChatID: {bool(self.chat_id)}")
        else:
            try:
                self.bot = Bot(token=self.bot_token)
                self.enabled = True
                logger.info("✅ Telegram Alert System initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")
                self.enabled = False

        # Alert queue for offline messages
        self.alert_queue: List[Alert] = []
        self.max_queue_size = 100

        # Alert filters
        self.min_priority = 1
        self.enabled_types = set(AlertType)

        # Rate limiting
        self.last_alert_time = datetime.now()
        self.min_alert_interval = 1  # seconds

        # Statistics
        self.alerts_sent = 0
        self.alerts_failed = 0

        # Storage
        self.data_dir = Path("10_DATA_STORAGE/alerts")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def send_alert_sync(self, alert: Alert) -> bool:
        """
        Send alert synchronously (wrapper for async method)
        """
        if not self.enabled:
            logger.warning("Telegram not enabled")
            return False

        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.send_alert(alert))
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Error in sync alert: {e}")
            return False

    async def send_alert(self, alert: Alert) -> bool:
        """
        Send alert to Telegram

        Args:
            alert: Alert object to send

        Returns:
            Success status
        """
        if not self.enabled:
            self._queue_alert(alert)
            logger.warning("Alert queued - Telegram not enabled")
            return False

        # Check filters
        if alert.priority < self.min_priority:
            logger.debug(f"Alert filtered by priority: {alert.title}")
            return False

        if alert.alert_type not in self.enabled_types:
            logger.debug(f"Alert type filtered: {alert.alert_type}")
            return False

        # Rate limiting
        if not self._check_rate_limit():
            self._queue_alert(alert)
            return False

        # Send alert
        try:
            message = alert.format_message()
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )

            self.alerts_sent += 1
            self._save_alert(alert, success=True)
            logger.info(f"✅ Alert sent: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            self.alerts_failed += 1
            self._queue_alert(alert)
            self._save_alert(alert, success=False, error=str(e))
            return False

    def send_trade_signal(self, symbol: str, action: str, price: float,
                         target: float, stop_loss: float, confidence: float = 0.0):
        """Send trade signal alert"""
        alert = Alert(
            alert_type=AlertType.TRADE_SIGNAL,
            title=f"{action} Signal: {symbol}",
            message=f"New trading signal generated",
            data={
                "Symbol": symbol,
                "Action": action,
                "Entry Price": f"₹{price:.2f}",
                "Target": f"₹{target:.2f}",
                "Stop Loss": f"₹{stop_loss:.2f}",
                "Risk/Reward": f"{((target-price)/(price-stop_loss)):.2f}",
                "Confidence": f"{confidence*100:.1f}%"
            },
            priority=3 if confidence > 0.7 else 2
        )

        return self.send_alert_sync(alert)

    def send_trade_executed(self, order_id: str, symbol: str, side: str,
                          quantity: int, price: float):
        """Send trade execution alert"""
        alert = Alert(
            alert_type=AlertType.TRADE_EXECUTED,
            title=f"Trade Executed: {symbol}",
            message=f"Order {order_id} has been filled",
            data={
                "Symbol": symbol,
                "Side": side,
                "Quantity": quantity,
                "Price": f"₹{price:.2f}",
                "Value": f"₹{price * quantity:,.2f}"
            },
            priority=2
        )

        return self.send_alert_sync(alert)

    def send_risk_warning(self, message: str, details: Dict):
        """Send risk warning alert"""
        alert = Alert(
            alert_type=AlertType.RISK_WARNING,
            title="Risk Alert",
            message=message,
            data=details,
            priority=3
        )

        return self.send_alert_sync(alert)

    def send_system_status(self, status: str, details: Optional[Dict] = None):
        """Send system status alert"""
        alert = Alert(
            alert_type=AlertType.SYSTEM_STATUS,
            title="System Status Update",
            message=status,
            data=details,
            priority=1
        )

        return self.send_alert_sync(alert)

    def send_test_message(self):
        """Send a test message to verify setup"""
        alert = Alert(
            alert_type=AlertType.INFO,
            title="MarketPulse Test Message",
            message="Your Telegram alerts are working correctly!",
            data={
                "System": "MarketPulse Trading",
                "Status": "Connected",
                "Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            priority=1
        )

        success = self.send_alert_sync(alert)
        if success:
            print("✅ Test message sent successfully! Check your Telegram.")
        else:
            print("❌ Failed to send test message. Check your credentials.")
        return success

    def _queue_alert(self, alert: Alert):
        """Queue alert for later sending"""
        if len(self.alert_queue) >= self.max_queue_size:
            self.alert_queue.pop(0)  # Remove oldest

        self.alert_queue.append(alert)
        logger.debug(f"Alert queued: {alert.title}")

    def _check_rate_limit(self) -> bool:
        """Check if enough time has passed since last alert"""
        now = datetime.now()
        time_diff = (now - self.last_alert_time).total_seconds()

        if time_diff < self.min_alert_interval:
            return False

        self.last_alert_time = now
        return True

    def _save_alert(self, alert: Alert, success: bool, error: Optional[str] = None):
        """Save alert to history file"""
        record = {
            'timestamp': alert.timestamp.isoformat(),
            'type': alert.alert_type.value,
            'title': alert.title,
            'message': alert.message,
            'data': alert.data,
            'priority': alert.priority,
            'success': success,
            'error': error
        }

        # Save to daily file
        date_str = alert.timestamp.strftime('%Y%m%d')
        file_path = self.data_dir / f"alerts_{date_str}.json"

        # Load existing or create new
        if file_path.exists():
            with open(file_path, 'r') as f:
                alerts = json.load(f)
        else:
            alerts = []

        alerts.append(record)

        # Save back
        with open(file_path, 'w') as f:
            json.dump(alerts, f, indent=2)

    def get_statistics(self) -> Dict:
        """Get alert system statistics"""
        return {
            'enabled': self.enabled,
            'alerts_sent': self.alerts_sent,
            'alerts_failed': self.alerts_failed,
            'queue_size': len(self.alert_queue),
            'min_priority': self.min_priority,
            'enabled_types': [t.value for t in self.enabled_types]
        }


def test_telegram_alerts():
    """Test Telegram alerts with better debugging"""
    print("\n=== Testing Telegram Alerts ===\n")

    # Show environment variables status
    print("Environment Check:")
    print(f"  TELEGRAM_BOT_TOKEN: {'✅ Set' if os.getenv('TELEGRAM_BOT_TOKEN') else '❌ Not set'}")
    print(f"  TELEGRAM_CHAT_ID: {'✅ Set' if os.getenv('TELEGRAM_CHAT_ID') else '❌ Not set'}")

    # Initialize alert system
    alert_system = TelegramAlertSystem()

    print(f"\nSystem Status:")
    print(f"  Enabled: {'✅ Yes' if alert_system.enabled else '❌ No'}")
    print(f"  Bot configured: {'✅ Yes' if alert_system.bot else '❌ No'}")

    if not alert_system.enabled:
        print("\n❌ Telegram not configured properly.")
        print("\nTroubleshooting:")
        print("1. Check your .env file has the correct format:")
        print("   TELEGRAM_BOT_TOKEN=your_token_here")
        print("   TELEGRAM_CHAT_ID=your_chat_id_here")
        print("2. Ensure python-telegram-bot is installed:")
        print("   pip install python-telegram-bot")
        return False

    # Send test message
    print("\nSending test message...")
    success = alert_system.send_test_message()

    if success:
        # Send additional test alerts
        print("\nSending sample alerts...")

        # Trade signal
        alert_system.send_trade_signal(
            symbol="RELIANCE",
            action="BUY",
            price=2500.0,
            target=2550.0,
            stop_loss=2475.0,
            confidence=0.75
        )
        print("  ✅ Trade signal sent")

        # Risk warning
        alert_system.send_risk_warning(
            message="Daily loss limit approaching",
            details={
                "Current Loss": "₹3,500",
                "Limit": "₹4,000",
                "Remaining": "₹500"
            }
        )
        print("  ✅ Risk warning sent")

        # System status
        alert_system.send_system_status(
            status="MarketPulse started successfully",
            details={
                "Mode": "Paper Trading",
                "Capital": "₹1,00,000",
                "ML Models": "Loaded"
            }
        )
        print("  ✅ System status sent")

        # Get statistics
        stats = alert_system.get_statistics()
        print(f"\nStatistics:")
        print(f"  Alerts sent: {stats['alerts_sent']}")
        print(f"  Alerts failed: {stats['alerts_failed']}")
        print(f"  Queue size: {stats['queue_size']}")

        print("\n✅ All test alerts sent! Check your Telegram.")

    return success


if __name__ == "__main__":
    # Test the system
    test_telegram_alerts()